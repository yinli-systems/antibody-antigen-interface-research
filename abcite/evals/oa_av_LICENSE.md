# OA audio / video eval licenses

AbCite OA AV evals use **already-public** clips. Media files (wav / ogg / mp4 /
keyframes) are **gitignored**. This file documents licenses for the candidate
lists in `evals/download_av.py`. Redistribute only under the listed terms.

Do not mix this gallery with the synthetic wav-hash / hue-keyframe fixtures.

Named `oa_av_LICENSE.md` (not `evals/av/`) so `python evals/run_*.py` does not
shadow PyAV (`import av`) via `evals/` on `sys.path`.

## Sources

- **Wikimedia Commons** — CC BY / CC BY-SA / CC0 / public domain / GFDL as
  recorded on each file’s Commons page (`imageinfo` LicenseShortName, 2026-09-10).
- **NASA images** (`images-assets.nasa.gov` `~small.mp4`) — NASA media, public
  domain (US government work). See NASA Media Usage Guidelines.

Fetcher: `python evals/download_av.py --max 64` then `--oa` evals.
Transcripts stay generic lecture filler unless `ABCITE_WHISPER=1`.

## Audio (Commons)

| id | file | license |
|---|---|---|
| oa-aud-heartbeat | File:Heartbeat.ogg | CC BY 3.0 |
| oa-aud-geiger | File:Geiger_counter.ogg | CC BY 3.0 |
| oa-aud-noise | File:White_noise.ogg | Public domain |
| oa-aud-thunder | File:Thunder.ogg | CC BY 3.0 |
| oa-aud-doppler | File:Fetal doppler recording heartbeat and movement.ogg | CC BY-SA 3.0 |
| oa-aud-korotkoff | File:Nl-Korotkoff-tonen.ogg | CC BY-SA 3.0 |
| oa-aud-cricket | File:Cricket.ogg | CC BY-SA 3.0 |
| oa-aud-siren | File:Siren.ogg | Public domain |
| oa-aud-a440 | File:440 Hz.ogg | Public domain |
| oa-aud-cat | File:Cat.ogg | CC BY 3.0 |
| oa-aud-bark | File:Barking.ogg | CC BY-SA 4.0 |
| oa-aud-frog | File:Frog.ogg | CC BY-SA 3.0 |
| oa-aud-applause | File:Applause.ogg | CC BY-SA 3.0 |
| oa-aud-bells | File:Church bells.ogg | Public domain |
| oa-aud-fire | File:Fire.ogg | Public domain |
| oa-aud-laugh | File:Laughter.ogg | Public domain |
| oa-aud-rain | File:Rain.ogg | Public domain |
| oa-aud-sneeze | File:Sneeze.ogg | CC0 |
| oa-aud-banjo | File:Banjo.ogg | CC BY-SA 3.0 |
| oa-aud-clock | File:Clock ticking.ogg | Public domain |
| oa-aud-flute | File:Flute.ogg | Public domain |
| oa-aud-harmonica | File:Harmonica.ogg | Public domain |
| oa-aud-piano | File:Piano.ogg | CC BY-SA 3.0 |
| oa-aud-brown | File:Brown noise.ogg | CC BY-SA 3.0 |
| oa-aud-pink | File:Pink noise.ogg | CC BY-SA 3.0 |
| oa-aud-morse | File:Morse code.ogg | Public domain |
| oa-aud-waterfall | File:Waterfall.ogg | CC BY-SA 4.0 |
| oa-aud-kick | File:Bass drum.ogg | CC BY-SA 3.0 |
| oa-aud-cicada | File:Cicada.ogg | CC BY-SA 3.0 |
| oa-aud-register | File:Cash register.ogg | Public domain |

## Video (Commons + NASA)

| id | file | license | source |
|---|---|---|---|
| oa-vid-gelbox | File:Conducting-Polymer-Electrodes-for-Gel-Electrophoresis-pone.0089416.s001.ogv | CC BY 4.0 | wikimedia-commons |
| oa-vid-fige | File:Increase-in-local-protein-concentration-by-field-inversion-gel-electrophoresis-1477-5956-5-18-S2.ogv | CC BY 2.0 | wikimedia-commons |
| oa-vid-anim | File:Electrophoresis explained - biology animation.webm | CC BY 3.0 | wikimedia-commons |
| oa-vid-confocal | File:Fluorescent and confocal microscopes.ogv | CC BY-SA 4.0 | wikimedia-commons |
| oa-vid-milk | File:Milk under the microscope.webm | CC BY 4.0 | wikimedia-commons |
| oa-vid-tem | File:Microscope Electronique à Transmission.ogv | CC BY-SA 4.0 | wikimedia-commons |
| oa-vid-neutrophil | File:Monomeric-and-Dimeric-CXCL8-Are-Both-Essential-for-In-Vivo-Neutrophil-Recruitment-pone.0011754.s004.ogv | CC BY 2.5 | wikimedia-commons |
| oa-vid-mitosis | File:Global-Analysis-of-CPEBs-Reveals-Sequential-and-Non-Redundant-Functions-in-Mitotic-Cell-Cycle-pone.0138794.s004.ogv | CC BY 4.0 | wikimedia-commons |
| oa-vid-polarize | File:A-Mathematical-Model-for-Neutrophil-Gradient-Sensing-and-Polarization-pcbi.0030036.sv002.ogv | CC BY 2.5 | wikimedia-commons |
| oa-vid-mitosis2 | File:Global-Analysis-of-CPEBs-Reveals-Sequential-and-Non-Redundant-Functions-in-Mitotic-Cell-Cycle-pone.0138794.s003.ogv | CC BY 4.0 | wikimedia-commons |
| oa-vid-amoeba | File:Amoeba proteus.ogv | CC BY-SA 3.0 | wikimedia-commons |
| oa-vid-cyclosis | File:Cyclosis.ogv | CC BY-SA 3.0 | wikimedia-commons |
| oa-vid-vessels | File:3-D-Visualization-and-Quantitation-of-Microvessels-in-Transparent-Human-Colorectal-Carcinoma-pone.0081857.s003.ogv | CC BY 2.5 | wikimedia-commons |
| oa-vid-microfluidic | File:A-Microfluidic-Platform-for-Correlative-Live-Cell-and-Super-Resolution-Microscopy-pone.0115512.s002.ogv | CC0 | wikimedia-commons |
| oa-vid-spheroid | File:A-High-Content-Small-Molecule-Screen-Identifies-Sensitivity-of-Glioblastoma-Stem-Cells-to-pone.0077053.s011.ogv | CC BY 2.5 | wikimedia-commons |
| oa-vid-filter | File:A-Microfiltration-Device-for-Urogenital-Schistosomiasis-Diagnostics-pone.0154640.s003.ogv | CC BY 4.0 | wikimedia-commons |
| oa-vid-ozone | File:2023 Ozone Hole Update.webm | Public domain | wikimedia-commons |
| oa-vid-cfd | File:Advanced Simulation Library - Multicomponent flow video.ogv | GFDL | wikimedia-commons |
| oa-vid-chestnut | File:Chestnut with maggot CT.ogv | CC BY-SA 3.0 | wikimedia-commons |
| oa-vid-earth | GSFC_20140421_EarthOrbit_m11525 | NASA public domain | nasa |
| oa-vid-launch | GSFC_20121101_GPM_m11129_Launch | NASA public domain | nasa |
| oa-vid-solar | GSFC_20160426_SDO_m12224_SolarFlare | NASA public domain | nasa |
| oa-vid-jupiter | GSFC_20180313_Jupiter_m12878_GreatRedSpot | NASA public domain | nasa |
| oa-vid-aurora | GSFC_20180314_Aurora_m12865_Steve | NASA public domain | nasa |
| oa-vid-rocket | KSC-20240525-MH-RKL01-0001-Rocket_Lab_PREFIRE_1_Launch_1080p-M6988 | NASA public domain | nasa |
| oa-vid-volcano | GSFC_20141113_Volcanoes_m10183_Degassing | NASA public domain | nasa |
