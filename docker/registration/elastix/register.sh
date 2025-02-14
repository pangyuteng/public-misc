#!/bin/bash
# bash register.sh sample_images/tlc.nii.gz sample_images/rv.nii.gz

export MOVING_NIFTI=$1 # TLC
export FIXED_NIFTI=$2 # RV

export AFFINE_DIR=outdir0
export SPLINEDIR1=outdir1
export SPLINEDIR2=outdir2

if [ ! -d "$AFFINE_DIR" ]; then
mkdir -p $AFFINE_DIR
elastix -f $FIXED_NIFTI -m $MOVING_NIFTI \
    -out $AFFINE_DIR \
    -p param/affine.txt
fi

if [ ! -d "$SPLINEDIR1" ]; then
mkdir -p $SPLINEDIR1
elastix -f $FIXED_NIFTI -m $MOVING_NIFTI \
    -out $SPLINEDIR1 \
    -p param/bspline1.txt -t0 $AFFINE_DIR/TransformParameters.0.txt
fi

if [ ! -d "$SPLINEDIR2" ]; then
mkdir -p $SPLINEDIR2
elastix -f $FIXED_NIFTI -m $MOVING_NIFTI \
    -out $SPLINEDIR2 \
    -p param/bspline2.txt -t0 $SPLINEDIR1/TransformParameters.0.txt    
fi

