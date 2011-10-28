Unsupervised joint alignment of complex images
http://vis-www.cs.umass.edu/code/congealingcomplex/

All code provided under a BSD-style license.  Terms of license can be
found at the top of each source file.


README contents:
--------------------------------
1. Overview
2. Quick Guide
3. Cars
4. Full Details
5. Additional Notes


1. Overview
--------------------------------

The source code is given for congealReal.cpp and funnelReal.cpp,
producing binaries congealReal and funnelReal.  Both are written in
C++ and require the OpenCV library (http://opencvlibrary.sourceforge.net).  

On a Linux machine, the following Makefile commands will produce the
binaries, following the conventions from
http://opencvlibrary.sourceforge.net/CompileOpenCVUsingLinux (refer to
the first URL for other environments).

IFLAGS = `pkg-config --cflags opencv` -O2
LFLAGS = `pkg-config --libs opencv`

all: congealReal funnelReal

congealReal: congealReal.cpp
	gcc $(IFLAGS) $(LFLAGS) -o congealReal congealReal.cpp

funnelReal: funnelReal.cpp
	gcc $(IFLAGS) $(LFLAGS) -o funnelReal funnelReal.cpp

Depending on the settings congealReal is run with, it may bring up
images in new windows.  Press any key to continue with the program.


2. Quick Guide
--------------------------------

congealReal images.list images.model

This command will read in a list of image filenames, one per line,
from images.list, perform congealing, and save the sequence of
distribution fields to the file images.model.  

funnelReal images.list images.model images_aligned.list

This command will read in a list of image filenames, one per line,
from images.list, align each image by funneling it according to the
sequence of distribution fields in images.model, then save the aligned
images using the filenames specified in images_aligned.list (which
should be in the same order as images.list).


3. Cars
--------------------------------

The set of car images used in the ICCV paper is provided to try with
the source code.

After uncompressing cars.tgz, one can run the following commands from
the cars/ directory.

congealReal carsFn.txt cars.train -o animSeq.txt -d display -v visualize -g carsOutFn.txt -outer 176 132 -inner 120 76 -nonrand -verbose

This will display the resulting images in new windows and save these
to the directory display, produce visualizations of the final
distribution field entropy and highest posterior probability cluster
representatives to visualize, generate aligned images and save them to
the directory final, use a 176x132 outer image and 120x76 inner image,
compute the entropy at all points within the inner window, and output
the entropy at each iteration.

congealReal carsFn.txt cars.train -a animSeq.txt animations -outer 176 132 -inner 120 76

This will produce frames for animation of the congealing done by the
previous command, saving the images grouped in 5x5 panels and entropy
of the distribution field, at each iteration of congealing, to the
directory animations.

funnelReal carsFn.txt cars.train carsOut.fn -outer 176 132 -inner 120 76 -o params.txt

This will align the car images using the funnel learned from the
congealing and save the aligned images to the directory final and the
transformation parameters used to align each image to params.txt.  

In the case of the cars, the funneling is duplicating the result of
congealing with the -g option, so this is just for illustration.
Funneling would be used in instances where it is not feasible to
congeal all images in a data set at once, or when congealing has been
done on an initial set of images and subsequently new images are found
that need to be aligned.


4. Full Details
--------------------------------

congealReal.cpp : congealing for complex, realistic images
                  using soft clusters of SIFT descriptors

usage : congealReal <list of image filenames> <model output file> ...
           [options]
        
        <list of image filenames> is a list of filenames of the images
           to process
        <model output file> is the filename to which the sequence of
           distribution fields should be written to (for use later in
           funneling)

        options :

           -o filename 
              output the transformations at each iteration to the
              specified file, in order to create an animation later

           -a filename directory
              create a frame (for animation) using the transformations
              given in the specified file, and write the result to the
              specified directory (must be used alone, and no
              congealing will be done)

           -v directory
              create visualizations of highest probability patches for
              each cluster and of entropy of final distribution field,
              writing images to the specified directory

           -g directory or list of filenames
              generate the final aligned images.  if the argument is a
              directory name, the images will be written to the
              specified directory using the original filenames (this
              assumes the original filenames were relative filenames,
              and appends them to the specified directory).
              otherwise, it is assumed the argument is the name of a
              file containing a list of filenames to use for the
              aligned images

           -d directory
              display the final transformations in 5x5 panels and
              write images to specified directory (press ESC to skip
              display of panels)

           -outer w h
              resize images to w by h for congealing computations
              (default 150x150)

           -inner w h
              use an inner window of size w by h, within which to
              calculate likelihood for congealing (must be smaller
              than outer dimensions by at least the size of the window
              for which SIFT descriptor is calculated over) (default
              100x100)

           -loc n
              sample n pixel locations at which to calculate
              likelihood for congealing (default 6,000)

           -nonrand
              use all points within inner window rather than sampling
              (will ignore -loc if provided)

           -clusters k
              use k clusters of SIFT descriptors (default 12)

           -verbose
              print out entropy for each iteration of congealing


funnelReal.cpp : funneling for complex, realistic images
                 using sequence of distribution fields learned from congealReal

usage : funnelReal <list of image filenames> ...
        <model file from congealing> ...
        <output directory or list of output filenames> [options]

        <list of image filenames> is a list of filenames of the images
           to process
        <model file from congealing> is the file containing the
           sequence of distribution fields from congealing
        <output directory or list of output filenames> if this is the
           name of a directory, the aligned images will be written to
           this directory (making the assumption to the filenames
           provided in the first argument are relative.  If it is not
           the name of the directory, then it should be the name of a
           file containing the filenames to use for the aligned
           images, in order corresponding to the first argument

        options :

           -o filename
              output the final parameter values used to generate
              aligned images

           -outer w h
              resize images to w by h for funneling computations
              (default 150x150).  this must match the values used in
              congealing

           -inner w h
              use an inner window of size w by h, within which to
              calculate likelihood for congealing (must be smaller
              than outer dimensions by at least the size of the window
              for which SIFT descriptor is calculated over) (default
              100x100).  this must match the values used in congealing


5. Additional Notes
--------------------------------

Currently, the SIFT descriptor computation remains as described in the
ICCV paper.  Important points are that it is computed over 8x8 patches
split into 4x4 subregions, yielding a 32 dimensional vector, and the
patches are not re-oriented to the dominent edge orientation of the
patch.

congealReal.cpp contains a constant variable maxIters = 100, and will
terminate if the number of iterations of congealing exceeds this
number.  In practice, congealing takes approximately 20 to 30
iterations.  Of course, this will vary depending on your particular
data, so you may wish to increase this number of necessary.

congealReal.cpp also contains a constant variable maxFrameIndex = 5.
This number determines how many sets of 5x5 images it will create when
making frames for animation.

funnelReal.cpp contains a constant variable maxProcessAtOnce = 600,
and will attempt to simultaneously funnel at most maxProcessAtOnce
images together.  This number should be set based on memory
constraints, though there should not be any significant slowdown if a
smaller number of images are funneled together in one round.

