// Exhaustive CUDA fixed-IMGT 90/90/90 search for one authenticated OAS shard.
//
// Candidate generation is lossless for the frozen 200-position metric.  The
// 200 positions are partitioned into 25 disjoint eight-byte signatures.  Any
// pair with at most 20 disagreements must share at least five signatures.  A
// dense uint8 target-by-query counter is inexpensive for the 3,103 AsEP
// development queries and avoids a capacity-limited per-target candidate map.
// Every quorum candidate is rechecked over all 200 positions with integer
// identity and reciprocal-coverage comparisons.

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t kQueryRecordBytes = 239;
constexpr std::size_t kTargetRecordBytes = 243;
constexpr std::size_t kAlignmentBytes = 200;
constexpr std::size_t kQueryAlignmentOffset = 39;
constexpr std::size_t kTargetAlignmentOffset = 43;
constexpr int kSeedGroups = 25;
constexpr int kSeedWidth = 8;
constexpr std::uint8_t kGap = 0;
constexpr std::uint8_t kMissingEnd = static_cast<std::uint8_t>('|');
constexpr std::uint16_t kEmptyTag = 0xffff;
constexpr int kThreads = 256;

struct Arguments {
  std::string query_path;
  std::string target_path;
  std::string match_path;
  std::string stats_path;
  std::size_t batch_records = 32768;
  int device = 0;
};

struct Key {
  std::uint64_t signature;
  std::uint16_t tag;
  bool operator==(const Key& other) const {
    return signature == other.signature && tag == other.tag;
  }
};

__host__ __device__ std::uint64_t splitmix64(std::uint64_t value) {
  value += 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

__host__ __device__ std::uint64_t key_hash(
    std::uint64_t signature, std::uint16_t tag) {
  return splitmix64(signature ^
                    (static_cast<std::uint64_t>(tag) *
                     0xd6e8feb86659fd93ULL));
}

struct KeyHash {
  std::size_t operator()(const Key& key) const {
    return static_cast<std::size_t>(key_hash(key.signature, key.tag));
  }
};

struct DeviceMatch {
  std::uint32_t target_index;
  std::uint16_t query_dense_index;
  std::uint16_t identical;
  std::uint16_t overlapping;
  std::uint16_t query_positions;
  std::uint16_t target_positions;
  std::uint16_t union_positions;
};

struct HostMatch {
  std::uint32_t query_ordinal;
  std::uint64_t target_ordinal;
  std::uint8_t target_split;
  std::uint16_t identical;
  std::uint16_t overlapping;
  std::uint16_t query_positions;
  std::uint16_t target_positions;
  std::uint16_t union_positions;
};

struct QueryIndex {
  std::vector<std::uint32_t> ordinals;
  std::vector<std::uint8_t> residue_counts;
  std::vector<std::uint8_t> alignments;
  std::vector<std::uint64_t> table_signatures;
  std::vector<std::uint16_t> table_tags;
  std::vector<std::uint32_t> table_offsets;
  std::vector<std::uint16_t> table_sizes;
  std::vector<std::uint16_t> bucket_members;
};

void cuda_check(cudaError_t status, const char* expression) {
  if (status != cudaSuccess) {
    throw std::runtime_error(
        std::string("CUDA failure at ") + expression + ": " +
        cudaGetErrorString(status));
  }
}

#define CUDA_CHECK(expr) cuda_check((expr), #expr)

std::uint32_t read_be32(const std::uint8_t* data) {
  return (static_cast<std::uint32_t>(data[0]) << 24) |
         (static_cast<std::uint32_t>(data[1]) << 16) |
         (static_cast<std::uint32_t>(data[2]) << 8) |
         static_cast<std::uint32_t>(data[3]);
}

std::uint64_t read_be64(const std::uint8_t* data) {
  std::uint64_t value = 0;
  for (int index = 0; index < 8; ++index) {
    value = (value << 8) | data[index];
  }
  return value;
}

void write_be16(std::ostream& output, std::uint16_t value) {
  const char bytes[2] = {
      static_cast<char>((value >> 8) & 0xff),
      static_cast<char>(value & 0xff),
  };
  output.write(bytes, 2);
}

void write_be32(std::ostream& output, std::uint32_t value) {
  const char bytes[4] = {
      static_cast<char>((value >> 24) & 0xff),
      static_cast<char>((value >> 16) & 0xff),
      static_cast<char>((value >> 8) & 0xff),
      static_cast<char>(value & 0xff),
  };
  output.write(bytes, 4);
}

void write_be64(std::ostream& output, std::uint64_t value) {
  char bytes[8];
  for (int index = 7; index >= 0; --index) {
    bytes[index] = static_cast<char>(value & 0xff);
    value >>= 8;
  }
  output.write(bytes, 8);
}

std::uint8_t normalize(std::uint8_t value) {
  return value == kMissingEnd ? kGap : value;
}

bool residue_byte(std::uint8_t value) {
  return value >= static_cast<std::uint8_t>('A') &&
         value <= static_cast<std::uint8_t>('Z');
}

std::uint16_t group_tag(int group) {
  if (group < 0 || group >= kSeedGroups) {
    throw std::runtime_error("Seed group is outside the frozen partition");
  }
  return static_cast<std::uint16_t>(group);
}

std::vector<std::uint8_t> read_file(const std::string& path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    throw std::runtime_error("Cannot open input file: " + path);
  }
  const std::streamsize size = input.tellg();
  if (size < 0) {
    throw std::runtime_error("Cannot size input file: " + path);
  }
  input.seekg(0);
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
  if (size && !input.read(reinterpret_cast<char*>(bytes.data()), size)) {
    throw std::runtime_error("Cannot read input file: " + path);
  }
  return bytes;
}

std::vector<std::array<int, kSeedWidth>> seed_positions() {
  std::vector<int> permutation;
  permutation.reserve(kAlignmentBytes);
  for (int index = 0; index < static_cast<int>(kAlignmentBytes); ++index) {
    permutation.push_back((19 + 73 * index) % static_cast<int>(kAlignmentBytes));
  }
  std::vector<std::array<int, kSeedWidth>> groups(kSeedGroups);
  for (int group = 0; group < kSeedGroups; ++group) {
    for (int member = 0; member < kSeedWidth; ++member) {
      groups[group][member] = permutation[group + member * kSeedGroups];
    }
    std::sort(groups[group].begin(), groups[group].end());
  }
  std::vector<int> flattened;
  for (const auto& group : groups) {
    flattened.insert(flattened.end(), group.begin(), group.end());
  }
  std::sort(flattened.begin(), flattened.end());
  for (int index = 0; index < static_cast<int>(kAlignmentBytes); ++index) {
    if (flattened[index] != index) {
      throw std::runtime_error("Seed partition is not exhaustive and disjoint");
    }
  }
  return groups;
}

std::uint64_t signature_for(
    const std::uint8_t* alignment,
    const std::array<int, kSeedWidth>& positions) {
  std::uint64_t signature = 0;
  for (int position : positions) {
    signature = (signature << 8) | normalize(alignment[position]);
  }
  return signature;
}

QueryIndex build_query_index(const std::string& path) {
  const auto bytes = read_file(path);
  if (bytes.empty() || bytes.size() % kQueryRecordBytes != 0) {
    throw std::runtime_error("Query binary size is not a positive record multiple");
  }
  const std::size_t query_count = bytes.size() / kQueryRecordBytes;
  if (query_count > std::numeric_limits<std::uint16_t>::max()) {
    throw std::runtime_error("Query count exceeds uint16 dense index");
  }
  const auto groups = seed_positions();
  std::unordered_map<Key, std::vector<std::uint16_t>, KeyHash> buckets;
  std::unordered_set<std::uint32_t> observed_ordinals;
  QueryIndex index;
  index.ordinals.reserve(query_count);
  index.residue_counts.reserve(query_count);
  index.alignments.reserve(query_count * kAlignmentBytes);
  for (std::size_t dense = 0; dense < query_count; ++dense) {
    const std::uint8_t* record = bytes.data() + dense * kQueryRecordBytes;
    const std::uint32_t ordinal = read_be32(record + 32);
    const std::uint8_t role = record[36];
    const std::uint8_t chain = record[37];
    const std::uint8_t claimed_count = record[38];
    const std::uint8_t* alignment = record + kQueryAlignmentOffset;
    if (role > 1 || chain < 1 || chain > 3 || claimed_count == 0 ||
        !observed_ordinals.insert(ordinal).second) {
      throw std::runtime_error("Query record metadata is invalid or duplicated");
    }
    int observed_count = 0;
    for (std::size_t position = 0; position < kAlignmentBytes; ++position) {
      const std::uint8_t value = alignment[position];
      if (value != kGap && value != kMissingEnd && !residue_byte(value)) {
        throw std::runtime_error("Query alignment contains an invalid byte");
      }
      observed_count += value != kGap && value != kMissingEnd;
    }
    if (observed_count != claimed_count) {
      throw std::runtime_error("Query residue count differs from alignment");
    }
    index.ordinals.push_back(ordinal);
    index.residue_counts.push_back(claimed_count);
    index.alignments.insert(
        index.alignments.end(), alignment, alignment + kAlignmentBytes);
    for (int group = 0; group < kSeedGroups; ++group) {
      // Chain type is authenticated metadata, not a search filter.  A
      // cross-labelled near-identical sequence must still be removed under the
      // conservative development decontamination policy.
      const Key key{
          signature_for(alignment, groups[group]), group_tag(group)};
      buckets[key].push_back(static_cast<std::uint16_t>(dense));
    }
  }

  std::size_t capacity = 1;
  while (capacity < buckets.size() * 4) {
    capacity <<= 1;
  }
  index.table_signatures.assign(capacity, 0);
  index.table_tags.assign(capacity, kEmptyTag);
  index.table_offsets.assign(capacity, 0);
  index.table_sizes.assign(capacity, 0);
  const std::size_t mask = capacity - 1;
  for (auto& item : buckets) {
    auto& members = item.second;
    std::sort(members.begin(), members.end());
    const std::uint32_t offset =
        static_cast<std::uint32_t>(index.bucket_members.size());
    index.bucket_members.insert(
        index.bucket_members.end(), members.begin(), members.end());
    std::size_t slot = key_hash(item.first.signature, item.first.tag) & mask;
    while (index.table_tags[slot] != kEmptyTag) {
      slot = (slot + 1) & mask;
    }
    index.table_signatures[slot] = item.first.signature;
    index.table_tags[slot] = item.first.tag;
    index.table_offsets[slot] = offset;
    index.table_sizes[slot] = static_cast<std::uint16_t>(members.size());
  }
  return index;
}

Arguments parse_arguments(int argc, char** argv) {
  Arguments arguments;
  for (int index = 1; index < argc; ++index) {
    const std::string key = argv[index];
    if (index + 1 >= argc) {
      throw std::runtime_error("Argument is missing a value: " + key);
    }
    const std::string value = argv[++index];
    if (key == "--queries") {
      arguments.query_path = value;
    } else if (key == "--targets") {
      arguments.target_path = value;
    } else if (key == "--matches") {
      arguments.match_path = value;
    } else if (key == "--stats") {
      arguments.stats_path = value;
    } else if (key == "--batch-records") {
      arguments.batch_records = std::stoull(value);
    } else if (key == "--device") {
      arguments.device = std::stoi(value);
    } else {
      throw std::runtime_error("Unknown argument: " + key);
    }
  }
  if (arguments.query_path.empty() || arguments.target_path.empty() ||
      arguments.match_path.empty() || arguments.stats_path.empty() ||
      arguments.batch_records < 1 || arguments.batch_records > 262144 ||
      arguments.device < 0) {
    throw std::runtime_error("Required path or bounded batch/device argument is invalid");
  }
  return arguments;
}

template <typename T>
T* copy_to_device(const std::vector<T>& source) {
  T* destination = nullptr;
  CUDA_CHECK(cudaMalloc(
      reinterpret_cast<void**>(&destination), source.size() * sizeof(T)));
  CUDA_CHECK(cudaMemcpy(
      destination,
      source.data(),
      source.size() * sizeof(T),
      cudaMemcpyHostToDevice));
  return destination;
}

__device__ std::uint8_t normalized_device(std::uint8_t value) {
  return value == kMissingEnd ? kGap : value;
}

__device__ std::uint16_t target_tag(int group) {
  return static_cast<std::uint16_t>(group);
}

__global__ void collect_seed_counts(
    const std::uint8_t* targets,
    std::size_t target_count,
    const int* positions,
    const std::uint64_t* table_signatures,
    const std::uint16_t* table_tags,
    const std::uint32_t* table_offsets,
    const std::uint16_t* table_sizes,
    std::size_t table_capacity,
    const std::uint16_t* bucket_members,
    const std::uint8_t* query_residue_counts,
    std::size_t query_count,
    std::uint8_t* counts,
    unsigned long long* bucket_member_visits,
    unsigned int* invalid_records) {
  const std::size_t target_index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (target_index >= target_count) {
    return;
  }
  const std::uint8_t* record = targets + target_index * kTargetRecordBytes;
  const std::uint8_t split = record[40];
  const std::uint8_t chain = record[41];
  const std::uint8_t claimed_residue_count = record[42];
  const std::uint8_t* alignment = record + kTargetAlignmentOffset;
  if (split > 1 || chain < 1 || chain > 3 || claimed_residue_count == 0) {
    atomicExch(invalid_records, 1U);
    return;
  }
  int observed_residue_count = 0;
  std::uint8_t* row = counts + target_index * query_count;
  const std::size_t mask = table_capacity - 1;
  for (int group = 0; group < kSeedGroups; ++group) {
    std::uint64_t signature = 0;
    for (int member = 0; member < kSeedWidth; ++member) {
      const std::uint8_t value = alignment[positions[group * kSeedWidth + member]];
      if (value != kGap && value != kMissingEnd) {
        if (value < static_cast<std::uint8_t>('A') ||
            value > static_cast<std::uint8_t>('Z')) {
          atomicExch(invalid_records, 1U);
        }
        ++observed_residue_count;
      }
      signature = (signature << 8) | normalized_device(value);
    }
    const std::uint16_t tag = target_tag(group);
    std::size_t slot = key_hash(signature, tag) & mask;
    while (true) {
      const std::uint16_t observed_tag = table_tags[slot];
      if (observed_tag == kEmptyTag) {
        break;
      }
      if (observed_tag == tag && table_signatures[slot] == signature) {
        const std::uint32_t offset = table_offsets[slot];
        const std::uint16_t size = table_sizes[slot];
        atomicAdd(bucket_member_visits, static_cast<unsigned long long>(size));
        for (std::uint16_t index = 0; index < size; ++index) {
          const std::uint16_t query = bucket_members[offset + index];
          const int query_count_value = query_residue_counts[query];
          if (10 * static_cast<int>(claimed_residue_count) >=
                  9 * query_count_value &&
              10 * query_count_value >=
                  9 * static_cast<int>(claimed_residue_count)) {
            ++row[query];
          }
        }
        break;
      }
      slot = (slot + 1) & mask;
    }
  }
  if (observed_residue_count != claimed_residue_count) {
    atomicExch(invalid_records, 1U);
  }
}

__device__ bool verify_pair(
    const std::uint8_t* query,
    const std::uint8_t* target,
    std::uint16_t* identical,
    std::uint16_t* overlapping,
    std::uint16_t* query_positions,
    std::uint16_t* target_positions,
    std::uint16_t* union_positions) {
  int same = 0;
  int overlap = 0;
  int query_count = 0;
  int target_count = 0;
  int either = 0;
  for (int position = 0; position < static_cast<int>(kAlignmentBytes); ++position) {
    const std::uint8_t query_value = query[position];
    const std::uint8_t target_value = target[position];
    const bool query_exists = query_value != kGap && query_value != kMissingEnd;
    const bool target_exists = target_value != kGap && target_value != kMissingEnd;
    query_count += query_exists;
    target_count += target_exists;
    overlap += query_exists && target_exists;
    either += query_exists || target_exists;
    same += query_exists && target_exists && query_value == target_value;
  }
  *identical = static_cast<std::uint16_t>(same);
  *overlapping = static_cast<std::uint16_t>(overlap);
  *query_positions = static_cast<std::uint16_t>(query_count);
  *target_positions = static_cast<std::uint16_t>(target_count);
  *union_positions = static_cast<std::uint16_t>(either);
  return 10 * same >= 9 * either &&
         10 * overlap >= 9 * query_count &&
         10 * overlap >= 9 * target_count;
}

__global__ void count_or_fill_matches(
    const std::uint8_t* targets,
    std::size_t target_count,
    const std::uint8_t* queries,
    std::size_t query_count,
    const std::uint8_t* counts,
    DeviceMatch* matches,
    unsigned long long* match_count,
    unsigned long long* quorum_candidates) {
  const std::size_t cell_count = target_count * query_count;
  for (std::size_t cell =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       cell < cell_count;
       cell += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    if (counts[cell] < 5) {
      continue;
    }
    const std::size_t target_index = cell / query_count;
    const std::size_t query_index = cell - target_index * query_count;
    if (matches == nullptr) {
      atomicAdd(quorum_candidates, 1ULL);
    }
    std::uint16_t identical = 0;
    std::uint16_t overlapping = 0;
    std::uint16_t query_positions = 0;
    std::uint16_t target_positions = 0;
    std::uint16_t union_positions = 0;
    if (!verify_pair(
            queries + query_index * kAlignmentBytes,
            targets + target_index * kTargetRecordBytes + kTargetAlignmentOffset,
            &identical,
            &overlapping,
            &query_positions,
            &target_positions,
            &union_positions)) {
      continue;
    }
    const unsigned long long output_index = atomicAdd(match_count, 1ULL);
    if (matches != nullptr) {
      matches[output_index] = DeviceMatch{
          static_cast<std::uint32_t>(target_index),
          static_cast<std::uint16_t>(query_index),
          identical,
          overlapping,
          query_positions,
          target_positions,
          union_positions,
      };
    }
  }
}

std::string json_escape(const std::string& value) {
  std::string escaped;
  for (char character : value) {
    if (character == '\\' || character == '"') {
      escaped.push_back('\\');
    }
    escaped.push_back(character);
  }
  return escaped;
}

void write_match(std::ostream& output, const HostMatch& match) {
  write_be32(output, match.query_ordinal);
  write_be64(output, match.target_ordinal);
  output.put(static_cast<char>(match.target_split));
  write_be16(output, match.identical);
  write_be16(output, match.overlapping);
  write_be16(output, match.query_positions);
  write_be16(output, match.target_positions);
  write_be16(output, match.union_positions);
}

int run(const Arguments& arguments) {
  CUDA_CHECK(cudaSetDevice(arguments.device));
  cudaDeviceProp properties{};
  CUDA_CHECK(cudaGetDeviceProperties(&properties, arguments.device));
  const auto query_index = build_query_index(arguments.query_path);
  const std::size_t query_count = query_index.ordinals.size();
  if (query_count == 0 || query_count > 4095) {
    throw std::runtime_error("Query count is outside the audited dense-search range");
  }

  std::ifstream targets(arguments.target_path, std::ios::binary | std::ios::ate);
  if (!targets) {
    throw std::runtime_error("Cannot open target binary");
  }
  const std::streamsize target_bytes_signed = targets.tellg();
  if (target_bytes_signed < 0 ||
      static_cast<std::uint64_t>(target_bytes_signed) % kTargetRecordBytes != 0) {
    throw std::runtime_error("Target binary size is not a record multiple");
  }
  const std::uint64_t target_bytes =
      static_cast<std::uint64_t>(target_bytes_signed);
  const std::uint64_t target_records = target_bytes / kTargetRecordBytes;
  targets.seekg(0);

  std::ofstream matches(arguments.match_path, std::ios::binary | std::ios::trunc);
  if (!matches) {
    throw std::runtime_error("Cannot create match binary");
  }

  const auto groups = seed_positions();
  std::vector<int> flat_positions;
  flat_positions.reserve(kAlignmentBytes);
  for (const auto& group : groups) {
    flat_positions.insert(flat_positions.end(), group.begin(), group.end());
  }

  int* device_positions = copy_to_device(flat_positions);
  std::uint64_t* device_table_signatures =
      copy_to_device(query_index.table_signatures);
  std::uint16_t* device_table_tags = copy_to_device(query_index.table_tags);
  std::uint32_t* device_table_offsets = copy_to_device(query_index.table_offsets);
  std::uint16_t* device_table_sizes = copy_to_device(query_index.table_sizes);
  std::uint16_t* device_bucket_members = copy_to_device(query_index.bucket_members);
  std::uint8_t* device_query_counts = copy_to_device(query_index.residue_counts);
  std::uint8_t* device_queries = copy_to_device(query_index.alignments);

  std::uint8_t* device_targets = nullptr;
  std::uint8_t* device_counts = nullptr;
  unsigned long long* device_bucket_visits = nullptr;
  unsigned long long* device_match_count = nullptr;
  unsigned long long* device_quorum_candidates = nullptr;
  unsigned int* device_invalid_records = nullptr;
  CUDA_CHECK(cudaMalloc(
      reinterpret_cast<void**>(&device_targets),
      arguments.batch_records * kTargetRecordBytes));
  CUDA_CHECK(cudaMalloc(
      reinterpret_cast<void**>(&device_counts),
      arguments.batch_records * query_count));
  CUDA_CHECK(cudaMalloc(
      reinterpret_cast<void**>(&device_bucket_visits),
      sizeof(unsigned long long)));
  CUDA_CHECK(cudaMalloc(
      reinterpret_cast<void**>(&device_match_count),
      sizeof(unsigned long long)));
  CUDA_CHECK(cudaMalloc(
      reinterpret_cast<void**>(&device_quorum_candidates),
      sizeof(unsigned long long)));
  CUDA_CHECK(cudaMalloc(
      reinterpret_cast<void**>(&device_invalid_records), sizeof(unsigned int)));
  CUDA_CHECK(cudaMemset(device_bucket_visits, 0, sizeof(unsigned long long)));
  CUDA_CHECK(cudaMemset(device_quorum_candidates, 0, sizeof(unsigned long long)));
  CUDA_CHECK(cudaMemset(device_invalid_records, 0, sizeof(unsigned int)));

  std::vector<std::uint8_t> host_targets(
      arguments.batch_records * kTargetRecordBytes);
  std::uint64_t records_scanned = 0;
  std::uint64_t total_matches = 0;
  std::uint64_t batch_count = 0;
  const auto start = std::chrono::steady_clock::now();
  while (records_scanned < target_records) {
    const std::size_t current = static_cast<std::size_t>(std::min<std::uint64_t>(
        arguments.batch_records, target_records - records_scanned));
    const std::size_t current_bytes = current * kTargetRecordBytes;
    if (!targets.read(
            reinterpret_cast<char*>(host_targets.data()),
            static_cast<std::streamsize>(current_bytes))) {
      throw std::runtime_error("Target binary ended before the authenticated size");
    }
    CUDA_CHECK(cudaMemcpy(
        device_targets,
        host_targets.data(),
        current_bytes,
        cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(device_counts, 0, current * query_count));
    CUDA_CHECK(cudaMemset(device_match_count, 0, sizeof(unsigned long long)));
    const int target_blocks = static_cast<int>((current + kThreads - 1) / kThreads);
    collect_seed_counts<<<target_blocks, kThreads>>>(
        device_targets,
        current,
        device_positions,
        device_table_signatures,
        device_table_tags,
        device_table_offsets,
        device_table_sizes,
        query_index.table_signatures.size(),
        device_bucket_members,
        device_query_counts,
        query_count,
        device_counts,
        device_bucket_visits,
        device_invalid_records);
    CUDA_CHECK(cudaGetLastError());
    const std::size_t cells = current * query_count;
    const int match_blocks = static_cast<int>(
        std::min<std::size_t>((cells + kThreads - 1) / kThreads, 65535));
    count_or_fill_matches<<<match_blocks, kThreads>>>(
        device_targets,
        current,
        device_queries,
        query_count,
        device_counts,
        nullptr,
        device_match_count,
        device_quorum_candidates);
    CUDA_CHECK(cudaGetLastError());
    unsigned long long current_matches = 0;
    CUDA_CHECK(cudaMemcpy(
        &current_matches,
        device_match_count,
        sizeof(current_matches),
        cudaMemcpyDeviceToHost));
    if (current_matches > 0) {
      DeviceMatch* device_matches = nullptr;
      CUDA_CHECK(cudaMalloc(
          reinterpret_cast<void**>(&device_matches),
          current_matches * sizeof(DeviceMatch)));
      CUDA_CHECK(cudaMemset(device_match_count, 0, sizeof(unsigned long long)));
      count_or_fill_matches<<<match_blocks, kThreads>>>(
          device_targets,
          current,
          device_queries,
          query_count,
          device_counts,
          device_matches,
          device_match_count,
          device_quorum_candidates);
      CUDA_CHECK(cudaGetLastError());
      std::vector<DeviceMatch> device_output(current_matches);
      CUDA_CHECK(cudaMemcpy(
          device_output.data(),
          device_matches,
          current_matches * sizeof(DeviceMatch),
          cudaMemcpyDeviceToHost));
      unsigned long long filled_matches = 0;
      CUDA_CHECK(cudaMemcpy(
          &filled_matches,
          device_match_count,
          sizeof(filled_matches),
          cudaMemcpyDeviceToHost));
      if (filled_matches != current_matches) {
        throw std::runtime_error("CUDA count and fill passes disagree");
      }
      CUDA_CHECK(cudaFree(device_matches));
      std::vector<HostMatch> host_output;
      host_output.reserve(current_matches);
      for (const auto& match : device_output) {
        if (match.target_index >= current ||
            match.query_dense_index >= query_count) {
          throw std::runtime_error("CUDA match index is outside the current batch");
        }
        const std::uint8_t* target =
            host_targets.data() + match.target_index * kTargetRecordBytes;
        host_output.push_back(HostMatch{
            query_index.ordinals[match.query_dense_index],
            read_be64(target + 32),
            target[40],
            match.identical,
            match.overlapping,
            match.query_positions,
            match.target_positions,
            match.union_positions,
        });
      }
      std::sort(
          host_output.begin(),
          host_output.end(),
          [](const HostMatch& left, const HostMatch& right) {
            return std::tie(left.target_ordinal, left.query_ordinal) <
                   std::tie(right.target_ordinal, right.query_ordinal);
          });
      for (const auto& match : host_output) {
        write_match(matches, match);
      }
      total_matches += current_matches;
    }
    records_scanned += current;
    ++batch_count;
  }
  matches.flush();
  if (!matches) {
    throw std::runtime_error("Cannot flush match binary");
  }
  unsigned int invalid_records = 0;
  unsigned long long bucket_visits = 0;
  unsigned long long quorum_candidates = 0;
  CUDA_CHECK(cudaMemcpy(
      &invalid_records,
      device_invalid_records,
      sizeof(invalid_records),
      cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(
      &bucket_visits,
      device_bucket_visits,
      sizeof(bucket_visits),
      cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(
      &quorum_candidates,
      device_quorum_candidates,
      sizeof(quorum_candidates),
      cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaDeviceSynchronize());
  if (invalid_records != 0) {
    throw std::runtime_error("At least one target record violated the binary schema");
  }
  const double elapsed = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - start).count();
  std::ofstream stats(arguments.stats_path, std::ios::binary | std::ios::trunc);
  if (!stats) {
    throw std::runtime_error("Cannot create kernel statistics file");
  }
  stats << "{\"batch_records\":" << arguments.batch_records
        << ",\"batches\":" << batch_count
        << ",\"bucket_member_visits\":" << bucket_visits
        << ",\"cuda_device\":\"" << json_escape(properties.name) << "\""
        << ",\"cuda_device_index\":" << arguments.device
        << ",\"elapsed_seconds\":" << std::setprecision(17) << elapsed
        << ",\"format_version\":\"oas_imgt_exact_cuda_stats_v1\""
        << ",\"matches\":" << total_matches
        << ",\"query_records\":" << query_count
        << ",\"quorum_candidates\":" << quorum_candidates
        << ",\"records_scanned\":" << records_scanned
        << ",\"seed_lookups\":" << records_scanned * kSeedGroups
        << ",\"target_file_records\":" << target_records
        << "}\n";
  stats.flush();
  if (!stats) {
    throw std::runtime_error("Cannot flush kernel statistics file");
  }
  CUDA_CHECK(cudaFree(device_invalid_records));
  CUDA_CHECK(cudaFree(device_quorum_candidates));
  CUDA_CHECK(cudaFree(device_match_count));
  CUDA_CHECK(cudaFree(device_bucket_visits));
  CUDA_CHECK(cudaFree(device_counts));
  CUDA_CHECK(cudaFree(device_targets));
  CUDA_CHECK(cudaFree(device_queries));
  CUDA_CHECK(cudaFree(device_query_counts));
  CUDA_CHECK(cudaFree(device_bucket_members));
  CUDA_CHECK(cudaFree(device_table_sizes));
  CUDA_CHECK(cudaFree(device_table_offsets));
  CUDA_CHECK(cudaFree(device_table_tags));
  CUDA_CHECK(cudaFree(device_table_signatures));
  CUDA_CHECK(cudaFree(device_positions));
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return run(parse_arguments(argc, argv));
  } catch (const std::exception& error) {
    std::cerr << "oas_imgt_exact_search: " << error.what() << '\n';
    return 1;
  }
}
