# Things we need to test

vvvvvvvvvvvvvvvvvvv Pre-calibration vvvvvvvvvvvvvvvvvvvvvvvv
Rosemary
* [x] pose calibration: 336
* RGB
  [x] nerfacto
  [x] iccv-4
* Main
  [x] iccv-4, wavelength_every=1: 309
  [x] iccv-4, wavelength_every=2: 308
  [x] iccv-4, wavelength_every=4: 324
  [x] iccv-4, wavelength_every=8: 325
* Ablation
  [x] iccv-1: 288
  [x] iccv-2: 290
  [x] iccv-3: 302
  [x] iccv-4: 309
  [x] iccv-5: 323
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


Rosemary
* [x] pose calibration: 336
* RGB
  [x] nerfacto 345
  [x] iccv-4 rgb: 373
  [x] iccv-2 rgb: 437
* Main
  [x] iccv-4, wavelength_every=1: 435, 440
  [ ] iccv-4, wavelength_every=2: 
  [ ] iccv-4, wavelength_every=4:
  [ ] iccv-4, wavelength_every=8: 
* Ablation
  [ ] iccv-1: 288
  [ ] iccv-2: 290
  [ ] iccv-3: 302
  [ ] iccv-4: 309
  [ ] iccv-5: 323


Basil:
* [x] pose calibration: 338
* RGB
  [x] nerfacto: 356
  [x] iccv-4 rgb: 363
  [x] iccv-2 rgb: 438
* Main
  [x] iccv-4, wavelength_every=1: 353
  [x] iccv-4, wavelength_every=2: 364
  [x] iccv-4, wavelength_every=4: 380
  [x] iccv-4, wavelength_every=8: 417
* Ablation
  [x] iccv-1: 366
  [x] iccv-2: 370
  [x] iccv-3: 372
  [ ] iccv-4: (353)
  [x] iccv-5: 419, cont as 425


<!-- Harsh1:
* [x] pose calibration: 388
* RGB
  [x] nerfacto: 390 -->
Harsh1 - Trying again:
* [.] pose calibration: 396
* RGB
  [x] nerfacto: 398
  [x] iccv-4 rgb: 399, 427, 429
  [x] iccv-2 rgb: 405
* Main
  [ ] iccv-4, wavelength_every=1: 434 ****, 444, 469
  [ ] iccv-4, wavelength_every=2:
  [ ] iccv-4, wavelength_every=4:
  [ ] iccv-4, wavelength_every=8:
* Ablation
  [x] iccv-1: 420
  [?] iccv-2: 421
  [x] iccv-3: 422
  [ ] iccv-4:
  [x] iccv-5: 433


<!-- Harsh2:
* [x] pose calibration: 371
* RGB
  [x] nerfacto: 376
  [ ] iccv-4 rgb:
* Main
  [.] iccv-4, wavelength_every=1: 392
  [ ] iccv-4, wavelength_every=2:
  [ ] iccv-4, wavelength_every=4:
  [ ] iccv-4, wavelength_every=8:
* Ablation
  [x] iccv-1: 385
  [.] iccv-2: 389
  [ ] iccv-3:
  [ ] iccv-4:
  [ ] iccv-5: -->

Harsh2 - trying again:
* [.] pose calibration: 394
* [.] pose calibration: 403
* RGB
  [x] nerfacto: 414
  [x] iccv-4 rgb: 413, (432?)
  [x] iccv-2 rgb: 416, 426, 428, 430, 431
* Main
  [x] iccv-4, wavelength_every=1: 451
  [ ] iccv-4, wavelength_every=2:
  [ ] iccv-4, wavelength_every=4:
  [ ] iccv-4, wavelength_every=8:
* Ablation
  [x] iccv-1: 385
  [ ] iccv-2:
  [ ] iccv-3:
  [ ] iccv-4:
  [ ] iccv-5:

```
set="rosemary_hs1"
shortname="rosemary"
set="basil_hs"
shortname="basil"
set="harsh_set1_new"
shortname="harsh1"
set="harsh_set2"
shortname="harsh2"

ns-train nerfacto-camera-pose-refinement --data $set --experiment-name pose_refinement_$shortname

# NEED TO RECALCULATE THE TRANSFORMS.JSON FILE

ns-train nerfacto --data $set --experiment-name refined-nerfacto-rgb-$shortname
ns-train hs-nerfacto3-rgb --data $set --experiment-name iccv4-rgb-$shortname
ns-train rgb-alpha-nerfacto --data $set --experiment-name iccv2-rgb-$shortname

ns-train iccv-4 --data $set --pipeline.model.train_wavelengths_every_nth 1 --experiment-name iccv4-ch1-main-$shortname
ns-train iccv-4 --data $set --pipeline.model.train_wavelengths_every_nth 2 --experiment-name iccv4-ch2-main-$shortname
ns-train iccv-4 --data $set --pipeline.model.train_wavelengths_every_nth 4 --experiment-name iccv4-ch4-main-$shortname
ns-train iccv-4 --data $set --pipeline.model.train_wavelengths_every_nth 8 --experiment-name iccv4-ch8-main-$shortname

ns-train iccv-1 --data $set --experiment-name iccv1-ablation-$shortname
ns-train iccv-2 --data $set --experiment-name iccv2-ablation-$shortname
ns-train iccv-3 --data $set --experiment-name iccv3-ablation-$shortname

ns-train iccv-5 --data $set --experiment-name iccv5-ablation-$shortname
```