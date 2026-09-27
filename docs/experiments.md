# Experimental evidence

This page records the experiments that were found on the development server and
that motivated the generic FPBA-Syn pipeline. The generated images and model
weights are not included in this repository. The numbers below are therefore a
reported reference run, not a claim that a fresh checkout reproduces every
artifact without access to the original data and checkpoints.

## Generation inventory

The recorded three-stage workflow is:

1. generate a scene-conditioned background with an EarthSynth ControlNet;
2. remove the source object with LaMa/inpainting;
3. redraw the foreground with the Flux Redux + Fill stage.

| Target category | Final images | Approx. source images | Variants/source |
| --- | ---: | ---: | ---: |
| Airplane | 20,472 | 6,824 | 3 |
| Launch vehicle / FSC | 1,455 | 485 | 3 |
| Ship | 2,805 | 935 | 3 |
| **Total** | **24,732** | **8,244** | |

Each large run retained both `bg/` and `final/` outputs, which is evidence that
the background-removal and redraw stages were both executed. An earlier small
end-to-end smoke test produced 15 final images from three source images.

## ControlNet fine-tuning

| Experiment | Steps | Batch / accumulation | Learning rate | Reported final loss |
| --- | ---: | --- | ---: | ---: |
| FSC / launch vehicle | 800 | 1 / 4 | 1e-5 | ~0.007 |
| Ship | 1,000 | 1 / 4 | 1e-5 | ~0.149 |

The loss is the noisy training scalar recorded by TensorBoard, not a validation
metric. For the FSC run, the loss ranged from about 0.77 at the beginning to a
minimum near 0.0016, with substantial stochastic variation. For the ship run,
the minimum was about 0.0003 and the final value was about 0.08--0.15 depending
on the final logged event. These values should not be interpreted as image
quality scores.

Additional historical runs included 5,000-step inpainting fine-tunes and a
500-step KC-10 LoRA experiment. They are retained as provenance only and are
not the recommended public benchmark.

## Detector-based object preservation

The following scores come from a class-matched greedy detector evaluation with
IoU >= 0.5. They measure whether generated targets remain detectable at the
expected locations; they are not human perceptual scores, FID, or a measure of
photorealism.

| Category | Images scored | Precision | Recall |
| --- | ---: | ---: | ---: |
| Airplane | 20,472 | 92.3% | 93.1% |
| Ship | 2,805 | 75.4% | 83.9% |
| FSC / launch vehicle | 1,401 | 83.5% | 83.4% |

The ship precision is the clearest quality weakness in this reference run and
should be improved before presenting the framework as production-ready.

## Downstream detection

On one recorded validation setup, a detector pretrained on the filtered
synthetic set reached COCO bbox mAP **0.683** (mAP50 **0.932**, mAP75 **0.835**).
Fine-tuning that model on real data reached mAP **0.739** (mAP50 **0.948**,
mAP75 **0.889**) at epoch 5. The validation split and configuration differ from
some other historical runs, so these values must not be compared directly with
the separate 0.760/0.863 real-data results.

## Reproducibility notes

- The airplane generation outputs still exist, but the corresponding airplane
  fine-tuning checkpoint was not present in the inspected server directory.
- The FSC preparation log reported 542 train + 61 validation images, while the
  currently retained conditioning directory contained 439 train + 49
  validation images. The data snapshot changed during the experiments.
- One airplane manifest contained fewer records than the number of final PNGs,
  so manifests should be regenerated and checked before release.
- Model weights, source datasets, server paths, credentials, and generated
  outputs are intentionally excluded from this repository.

## Suggested headline for the project page

> FPBA-Syn generated 24.7k remote-sensing images across airplanes, ships, and
> launch vehicles using a configurable ControlNet → background inpainting →
> Flux redraw pipeline. In a detector-based IoU@0.5 audit, target recall was
> 93.1% / 83.9% / 83.4% for airplanes / ships / FSC, respectively. Synthetic
> pretraining reached 0.683 bbox mAP before real-data fine-tuning.

