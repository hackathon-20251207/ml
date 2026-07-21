# Third-party notices

Sigma Sign's ML service uses model and dataset material created by other
projects. This file records provenance and attribution; it is not a substitute
for the complete upstream license texts.

## Easy Sign model and label mapping

- **Work:** Easy Sign Russian Sign Language recognition model and label mapping
- **Source:** [ai-forever/easy_sign](https://github.com/ai-forever/easy_sign)
- **Upstream license:** [Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)](https://github.com/ai-forever/easy_sign/blob/main/LICENSE)
- **Use in Sigma Sign:** isolated-gesture inference through ONNX Runtime

The production artifacts match the upstream files byte-for-byte:

| File | SHA-256 |
| --- | --- |
| `S3D.onnx` | `860ecb5e5aff91b4709016c2dc4f5744eea53e024f80c0b3b8f0f916f6bdb949` |
| `RSL_class_list.txt` | `390e90884aeac96c03ef6db87754ea62cb15b4a5b58f3659a5a900153e97f672` |

Easy Sign reports that its S3D model was trained on approximately 180,000
gesture examples, approximately 20,000 of them from Slovo, and recognizes
1,598 RSL gestures. The label mapping has 1,599 entries including `no`.
Sigma Sign did not train or adapt this model and does not claim authorship
of either artifact.

The ONNX model is not committed to this repository. Production downloads it
from configured S3-compatible storage, and the regression evaluator downloads
the pinned upstream artifact into a local ignored cache. The upstream label
mapping is present at `offline_inference/RSL_class_list.txt` so that class IDs
can be interpreted consistently.

## Slovo dataset

- **Work:** Slovo: Russian Sign Language Dataset
- **Authors:** Alexander Kapitanov, Karina Kvanchiani, Alexander Nagaev and Elizaveta Petrova
- **Source:** [hukenovs/slovo](https://github.com/hukenovs/slovo)
- **Paper:** [Slovo: Russian Sign Language Dataset](https://arxiv.org/abs/2305.14527)
- **Upstream license:** a [CC BY-SA 4.0 license variant](https://github.com/hukenovs/slovo/blob/master/license/en_us.pdf), as specified by the dataset authors

Slovo contains 20,400 videos across 1,001 classes including a no-event class,
recorded by 194 signers. It contributes to Easy Sign's training data but does
not define the complete Easy Sign vocabulary: 785 of the 999 named Slovo
glosses match the production label mapping by exact string.

`evaluation/slovo_golden.json` contains only a deterministic selection
manifest: official test-split video IDs, labels, byte sizes and SHA-256
checksums. Running `evaluation/evaluate_slovo.py` uses HTTP range requests to
download those 20 fixed videos from the official Slovo archive into a local
ignored cache. The videos themselves are not committed to this repository.
The selection and preprocessing do not change the copyright or license of the
source videos.

If results from the Slovo material are published, cite the dataset paper:

```bibtex
@inproceedings{kapitanov2023slovo,
  title        = {Slovo: Russian Sign Language Dataset},
  author       = {Kapitanov, Alexander and Kvanchiani, Karina and Nagaev, Alexander and Petrova, Elizaveta},
  booktitle    = {International Conference on Computer Vision Systems},
  pages        = {63--73},
  year         = {2023},
  organization = {Springer}
}
```

## Redistribution

CC BY-SA 4.0 requires attribution and, when licensed or adapted material is
shared, distribution under the same or a compatible license. Preserve links
to the upstream works, their license notices, and an indication of changes.
Review the complete upstream license terms before redistributing the model,
mapping or videos.

No affiliation with or endorsement by AI Forever, the Easy Sign maintainers,
the Slovo authors or their institutions is implied.
