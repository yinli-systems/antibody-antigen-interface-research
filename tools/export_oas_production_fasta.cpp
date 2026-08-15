#include <algorithm>
#include <array>
#include <cerrno>
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

[[noreturn]] void fail(const std::string& message) {
  throw std::runtime_error(message);
}

std::vector<std::string> split_tabs(const std::string& line) {
  std::vector<std::string> fields;
  std::size_t start = 0;
  while (true) {
    const auto position = line.find('\t', start);
    fields.push_back(line.substr(start, position - start));
    if (position == std::string::npos) return fields;
    start = position + 1;
  }
}

std::uint64_t parse_uint64(const std::string& value, const char* label) {
  std::uint64_t output = 0;
  const auto result = std::from_chars(value.data(), value.data() + value.size(), output);
  if (result.ec != std::errc{} || result.ptr != value.data() + value.size()) {
    fail(std::string("Invalid ") + label);
  }
  return output;
}

class MappedFile {
 public:
  explicit MappedFile(const std::string& path) : path_(path) {
    descriptor_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (descriptor_ < 0) fail("Cannot open array: " + path);
    struct stat metadata {};
    if (::fstat(descriptor_, &metadata) != 0 || !S_ISREG(metadata.st_mode) || metadata.st_size <= 0) {
      fail("Array is not a nonempty regular file: " + path);
    }
    size_ = static_cast<std::size_t>(metadata.st_size);
    void* mapped = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, descriptor_, 0);
    if (mapped == MAP_FAILED) {
      fail("Cannot map array: " + path);
    }
    mapping_ = static_cast<const std::uint8_t*>(mapped);
  }

  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;

  ~MappedFile() {
    if (mapping_ != nullptr) ::munmap(const_cast<std::uint8_t*>(mapping_), size_);
    if (descriptor_ >= 0) ::close(descriptor_);
  }

  const std::uint8_t* data() const { return mapping_; }
  std::size_t size() const { return size_; }

 private:
  std::string path_;
  int descriptor_ = -1;
  const std::uint8_t* mapping_ = nullptr;
  std::size_t size_ = 0;
};

struct NpyArray {
  explicit NpyArray(const std::string& path) : file(path) {
    if (file.size() < 12 || std::memcmp(file.data(), "\x93NUMPY", 6) != 0) {
      fail("Array is not NPY: " + path);
    }
    const std::uint8_t major = file.data()[6];
    std::size_t header_length = 0;
    std::size_t header_offset = 0;
    if (major == 1) {
      header_length = static_cast<std::size_t>(file.data()[8]) |
                      (static_cast<std::size_t>(file.data()[9]) << 8);
      header_offset = 10;
    } else if (major == 2 || major == 3) {
      header_length = static_cast<std::size_t>(file.data()[8]) |
                      (static_cast<std::size_t>(file.data()[9]) << 8) |
                      (static_cast<std::size_t>(file.data()[10]) << 16) |
                      (static_cast<std::size_t>(file.data()[11]) << 24);
      header_offset = 12;
    } else {
      fail("Unsupported NPY version: " + path);
    }
    if (header_offset + header_length > file.size()) fail("Truncated NPY header: " + path);
    header.assign(reinterpret_cast<const char*>(file.data() + header_offset), header_length);
    data_offset = header_offset + header_length;
    if (header.find("'fortran_order': False") == std::string::npos) {
      fail("Fortran-order NPY is unsupported: " + path);
    }
    const auto shape_key = header.find("'shape':");
    const auto left = header.find('(', shape_key);
    const auto right = header.find(')', left);
    if (shape_key == std::string::npos || left == std::string::npos || right == std::string::npos) {
      fail("NPY shape is malformed: " + path);
    }
    std::string shape_text = header.substr(left + 1, right - left - 1);
    std::size_t start = 0;
    while (start < shape_text.size()) {
      while (start < shape_text.size() && (shape_text[start] == ' ' || shape_text[start] == ',')) ++start;
      if (start == shape_text.size()) break;
      std::size_t stop = start;
      while (stop < shape_text.size() && shape_text[stop] >= '0' && shape_text[stop] <= '9') ++stop;
      if (stop == start) fail("NPY dimension is malformed: " + path);
      shape.push_back(parse_uint64(shape_text.substr(start, stop - start), "NPY dimension"));
      start = stop;
    }
    if (shape.empty()) fail("NPY shape is empty: " + path);
  }

  bool has_dtype(const std::string& value) const {
    return header.find("'descr': '" + value + "'") != std::string::npos;
  }

  MappedFile file;
  std::string header;
  std::vector<std::uint64_t> shape;
  std::size_t data_offset = 0;
};

class BufferedWriter {
 public:
  explicit BufferedWriter(const std::string& path) : buffer_(16 * 1024 * 1024) {
    descriptor_ = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0444);
    if (descriptor_ < 0) fail("Cannot create FASTA output: " + path);
  }

  BufferedWriter(const BufferedWriter&) = delete;
  BufferedWriter& operator=(const BufferedWriter&) = delete;

  ~BufferedWriter() {
    try {
      flush();
    } catch (...) {
    }
    if (descriptor_ >= 0) ::close(descriptor_);
  }

  void append(const char* data, std::size_t size) {
    std::size_t offset = 0;
    while (offset < size) {
      if (used_ == buffer_.size()) flush();
      const std::size_t count = std::min(size - offset, buffer_.size() - used_);
      std::memcpy(buffer_.data() + used_, data + offset, count);
      used_ += count;
      offset += count;
    }
  }

  void append_char(char value) { append(&value, 1); }

  void flush() {
    std::size_t offset = 0;
    while (offset < used_) {
      const ssize_t count = ::write(descriptor_, buffer_.data() + offset, used_ - offset);
      if (count < 0) {
        if (errno == EINTR) continue;
        fail("Cannot write FASTA output");
      }
      offset += static_cast<std::size_t>(count);
    }
    used_ = 0;
  }

 private:
  int descriptor_ = -1;
  std::vector<char> buffer_;
  std::size_t used_ = 0;
};

struct Shard {
  std::string split;
  std::uint64_t bucket;
  std::uint64_t global_offset;
  std::uint64_t record_count;
  std::string tokens_path;
  std::string lengths_path;
};

std::vector<Shard> read_inventory(const std::string& path) {
  std::ifstream input(path);
  if (!input) fail("Cannot open exporter inventory");
  std::string line;
  if (!std::getline(input, line) ||
      line != "split\tbucket\tglobal_offset\trecord_count\ttokens_path\tlengths_path") {
    fail("Exporter inventory header differs");
  }
  std::vector<Shard> output;
  while (std::getline(input, line)) {
    if (line.empty()) fail("Exporter inventory contains a blank row");
    const auto fields = split_tabs(line);
    if (fields.size() != 6 || (fields[0] != "train" && fields[0] != "validation")) {
      fail("Exporter inventory row differs");
    }
    output.push_back(Shard{fields[0], parse_uint64(fields[1], "bucket"),
                           parse_uint64(fields[2], "global offset"),
                           parse_uint64(fields[3], "record count"), fields[4], fields[5]});
  }
  if (output.empty()) fail("Exporter inventory is empty");
  return output;
}

std::uint16_t read_u16_le(const std::uint8_t* data) {
  return static_cast<std::uint16_t>(data[0]) |
         (static_cast<std::uint16_t>(data[1]) << 8);
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 3) fail("Usage: export_oas_production_fasta INVENTORY OUTPUT_FASTA");
    const auto shards = read_inventory(argv[1]);
    BufferedWriter output(argv[2]);
    constexpr std::string_view residues = "ACDEFGHIKLMNPQRSTVWYX";
    std::array<std::uint64_t, 2> next_offset{0, 0};
    std::array<std::uint64_t, 2> record_counts{0, 0};
    std::array<std::uint64_t, 2> residue_counts{0, 0};
    for (const auto& shard : shards) {
      const std::size_t split_index = shard.split == "train" ? 0 : 1;
      if (shard.bucket >= 256 || shard.global_offset != next_offset[split_index]) {
        fail("Exporter shard order or offset differs");
      }
      NpyArray tokens(shard.tokens_path);
      NpyArray lengths(shard.lengths_path);
      if (!tokens.has_dtype("|u1") || tokens.shape.size() != 2 || tokens.shape[0] != shard.record_count ||
          tokens.shape[1] != 194 || tokens.data_offset + shard.record_count * 194 != tokens.file.size()) {
        fail("Token array schema differs");
      }
      if (!(lengths.has_dtype("<u2") || lengths.has_dtype("|u2")) || lengths.shape.size() != 1 ||
          lengths.shape[0] != shard.record_count || lengths.data_offset + shard.record_count * 2 != lengths.file.size()) {
        fail("Length array schema differs");
      }
      const auto* token_data = tokens.file.data() + tokens.data_offset;
      const auto* length_data = lengths.file.data() + lengths.data_offset;
      for (std::uint64_t row = 0; row < shard.record_count; ++row) {
        const std::uint16_t encoded_length = read_u16_le(length_data + row * 2);
        if (encoded_length < 82 || encoded_length > 194) fail("Encoded sequence length differs");
        const auto* sequence_tokens = token_data + row * 194;
        if (sequence_tokens[0] != 2 || sequence_tokens[encoded_length - 1] != 3) {
          fail("Sequence boundary token differs");
        }
        for (std::size_t index = encoded_length; index < 194; ++index) {
          if (sequence_tokens[index] != 0) fail("Sequence padding token differs");
        }
        char header[32];
        const char prefix = shard.split == "train" ? 't' : 'v';
        const auto identifier = shard.global_offset + row;
        const int header_size = std::snprintf(header, sizeof(header), ">%c%llu\n", prefix,
                                              static_cast<unsigned long long>(identifier));
        if (header_size <= 0 || static_cast<std::size_t>(header_size) >= sizeof(header)) {
          fail("FASTA identifier overflow");
        }
        output.append(header, static_cast<std::size_t>(header_size));
        for (std::uint16_t index = 1; index + 1 < encoded_length; ++index) {
          const std::uint8_t token = sequence_tokens[index];
          if (token < 5 || token >= 5 + residues.size()) fail("Residue token differs");
          output.append_char(residues[token - 5]);
        }
        output.append_char('\n');
        ++record_counts[split_index];
        residue_counts[split_index] += encoded_length - 2;
      }
      next_offset[split_index] += shard.record_count;
    }
    output.flush();
    std::cout << "train_records=" << record_counts[0] << "\n"
              << "validation_records=" << record_counts[1] << "\n"
              << "train_residues=" << residue_counts[0] << "\n"
              << "validation_residues=" << residue_counts[1] << "\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
}
