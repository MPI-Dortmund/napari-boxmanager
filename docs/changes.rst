Changes
=======

v0.3.16:
*******
  - Fix visibility issue when working with filtered images and multiple sets.  (https://github.com/MPI-Dortmund/napari-boxmanager/issues/16)

v0.3.15:
*******
  - Fix visibility issue when working with multiple tomograms / image sets. Thanks to Sreeparna Biswas (https://github.com/MPI-Dortmund/napari-boxmanager/issues/14)

v0.3.14:
*******
  - Fix problem when opening folder of tomograms (https://github.com/MPI-Dortmund/napari-boxmanager/issues/12)
  - Filament distance renamed to Inter-box distance

v0.3.13:
*******
  - Fix crash when writing box files introduced with 0.3.12 (Thanks to Jake D. Johnston)
  - Improves plotting in some edge cases


v0.3.12:
*******
 - Fix crash when readining cbox files (Thanks to and Jake D. Johnston and Yasunori Saitoh)
 - Written CBOX files now also contain the filament verticis.

v0.3.11:
*******
 - Fix crash when creating new layer (introduced with v0.3.7).

v0.3.10:
*******
 - Fix crash when creating new layer (introduced with v0.3.7).

v0.3.10:
*******
 - Fix reading Float16 mrc files.

v0.3.9:
*******
 - Fix cbox reading that got broken before.

v0.3.7:
*******
 - Linking now allows renaming
 - Filtered images now get automatically linked

v0.3.6:
*******
 - Fix error that occurs when filament and particle cbox files get loaded together.

v0.3.5:
*******
 - Fix error that occurs when coordinate layers are deleted and then reloaded.

v0.3.4:
*******
 - Make "Add" mode default when clicking on "Create Particle Layer"
 - Read/write support for filaments in CBOX format
 - Read/write support for filament in helicon format
 - Read/write support for Relion STAR files.
 - Option in organize_layer to save all coordinates/filaments to a directory
 - Add a simplified call 'napari_boxmanager'
 - Improved matching
 - Many bug fixes

v0.2.10:
********
 - Fix bug when reading .tloc files with long paths (Thanks Tom Dendooven)
 - Fix bug with contrast issues when low pass filter images
 - Add smooth fall-off when low pass filtering images
