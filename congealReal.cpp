// congealReal.cpp : congealing for complex, realistic images
//                   using soft clusters of SIFT descriptors

/*
* Copyright (c) 2007, Gary B. Huang, UMass-Amherst
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*     * Redistributions of source code must retain the above copyright
*       notice, this list of conditions and the following disclaimer.
*     * Redistributions in binary form must reproduce the above copyright
*       notice, this list of conditions and the following disclaimer in the
*       documentation and/or other materials provided with the distribution.
*     * Neither the name of the author nor the organization may be used to 
*       endorse or promote products derived from this software without specific 
*       prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY Gary B. Huang ``AS IS'' AND ANY
* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL <copyright holder> BE LIABLE FOR ANY
* DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

// usage : congealReal <list of image filenames> <model output file> [options]
//         
//         <list of image filenames> is a list of filenames of the images to process
//         <model output file> is the filename to which the sequence of distribution fields 
//            should be written to (for use later in funneling)
//
//         options :
//
//            -o filename 
//               output the transformations at each iteration to the specified file, 
//               in order to create an animation later
//
//            -a filename directory
//               create a frame (for animation) using the transformations given in the specified
//               file, and write the result to the specified directory
//                 (must be used alone, and no congealing will be done)
//
//            -v directory
//               create visualizations of highest probability patches for each cluster and of entropy
//               of final distribution field, writing images to the specified directory
//
//            -g directory or list of filenames
//               generate the final aligned images.  if the argument is a directory name, the images will be
//               written to the specified directory using the original filenames (this assumes the
//               original filenames were relative filenames, and appends them to the specified directory).
//               otherwise, it is assumed the argument is the name of a file containing a list of filenames 
//               to use for the aligned images
//
//            -d directory
//               display the final transformations in 5x5 panels and write images to specified directory
//               (press ESC to skip display of panels)
//
//            -outer w h
//               resize images to w by h for congealing computations (default 150x150)
//
//            -inner w h
//               use an inner window of size w by h, within which to calculate likelihood for congealing
//               (must be smaller than outer dimensions by at least the size of the window for which
//               SIFT descriptor is calculated over) (default 100x100)
//
//            -loc n
//               sample n pixel locations at which to calculate likelihood for congealing (default 6,000)
//
//            -nonrand
//               use all points within inner window rather than sampling (will ignore -loc if provided)
//
//            -clusters k
//               use k clusters of SIFT descriptors (default 12)
//
//            -verbose
//               print out entropy for each iteration of congealing

// dimensions used for original ICCV experiments :
//    faces : outer 148x148, inner 100x100
//    cars  : outer 176x132, inner 120x76

#include <cv.h>
#include <cxcore.h>
#include <highgui.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include "math.h"
#include "float.h"

using namespace std;

void MakeFrame(vector<IplImage *> &images, vector<vector<float> > &v, int h, int w, 
	       vector<vector<vector<float> > > &bSq, string basefn, int frameIndex, int maxFrameIndex);
void makeEntFrame(vector<vector<float> > &logDistField, vector<pair<int, int> > &randLocs, char *fn, float minEnt, float maxEnt,
		  int innerDimH, int innerDimW, int paddingH, int paddingW);
void showResults(char *imageListFn, vector<vector<float> > &v, int h, int w,
		 int paddingH, int paddingW, vector<pair<int, float> > &indexProbPairs,
		 bool display, string dDirectory, bool generateFinal, string gDirectory);
float computeLogLikelihood(vector<vector<float> > &distField, vector<vector<float> > &fids, int numFeatureClusters);
float computeEntropy(vector<vector<float> > &distField, int numFeatureClusters);
void getNewFeatsInvT(vector<vector<float> > &newFIDs, vector<vector<vector<float> > > &originalFeats, 
		     vector<float> &vparams, float centerX, float centerY, vector<pair<int, int> > &randLocs);
float dist(vector<float> &a, vector<float> &b);
void getSIFTdescripter(vector<float> &descripter, vector<vector<float> > &m, vector<vector<float> > &theta, int x, int y, int windowSize, int histDim, int bucketsDim, 
		       vector<vector<float> > &Gaussian);
float findPrincipalAngle(float a1, float v1, float a2, float v2, float a3, float v3);
void computeGaussian(vector<vector<float> > &Gaussian, int windowSize);
void setRandLocs(vector<pair<int, int> > &randLocs, int h, int w, int paddingH, int paddingW, bool nonRand);
void reorientM(vector<vector<float> > &m, vector<vector<float> > &newM, 
	       vector<vector<float> > &theta, vector<vector<float> > &newTheta,
	       float angle, float cx, float cy, int windowSize);

const float pi = (float)3.14159265;

bool indexProbComparison(const pair<int, float> p1, const pair<int, float> p2)
{
  return p1.second > p2.second;
}

int main(int argc, char* argv[])
{
  if(argc < 3)
    {
      cout << "usage: " << argv[0] << " <list of image filenames> <output file> [options]" << endl;
      return -1;
    }

  const int numParams = 4; // similarity transforms - x translation, y translation, rotation, uniform scaling
  const int maxFrameIndex = 5;
  const int maxIters = 100;

  int windowSize = 4; // half window size

  int argcIndex = 3;
  bool animation = false;
  bool outputParams = false;
  string oaFilename, aDirectory;
  bool visualize = false;
  string vDirectory;
  bool generateFinal = false;
  string gDirectory;
  bool verbose = false;

  bool display = false;
  string dDirectory;
  bool nonRand = false;

  int outerDimW = 150, outerDimH = 150, innerDimW = 100, innerDimH = 100;
  int numRandLoc = 6000;
  int numFeatureClusters = 12;

  // reading in options
  while(argcIndex < argc)
    {
      switch(argv[argcIndex][1])
	{
	case 'o':
	  {
	    if(!strcmp(argv[argcIndex], "-o"))
	      {
		if(argcIndex == argc - 1 || argv[argcIndex+1][0] == '-')
		  {
		    cout << "no output filename provided" << endl;
		    return -1;
		  }
		outputParams = true;
		oaFilename = argv[argcIndex+1];
		argcIndex += 2;	
	      }
	    else
	      {
		if(argcIndex >= argc - 2 || argv[argcIndex+1][0] == '-'
		   || argv[argcIndex+2][0] == '-')
		  {
		    cout << "not enough parameters specified" << endl;
		    return -1;
		  }
		outerDimW = atoi(argv[argcIndex+1]);
		outerDimH = atoi(argv[argcIndex+2]);
		argcIndex += 3;
	      }
	    break;
	  }
	case 'a':
	  {
	    cout << "just creating animation frames" << endl;
	    if(argcIndex >= argc - 2 || argv[argcIndex+1][0] == '-'
	       || argv[argcIndex+2][0] == '-')
	      {
		cout << "not enough parameters specified" << endl;
		return -1;
	      }
	    animation = true;
	    oaFilename = argv[argcIndex+1];
	    aDirectory = argv[argcIndex+2];
	    argcIndex += 3;
	    break;
	  }
	case 'v':
	  {
	    if(!strcmp(argv[argcIndex], "-v"))
	      {
		if(argcIndex == argc - 1 || argv[argcIndex+1][0] == '-')
		  {
		    cout << "no output directory provided" << endl;
		    return -1;
		  }
		visualize = true;
		vDirectory = argv[argcIndex+1];
		argcIndex += 2;
	      }
	    else
	      {
		verbose = true;
		++argcIndex;
	      }
	    break;
	  }
	case 'g':
	  {
	    if(argcIndex == argc - 1 || argv[argcIndex+1][0] == '-')
	      {
		cout << "no output directory provided" << endl;
		return -1;
	      }
	    generateFinal = true;
	    gDirectory = argv[argcIndex+1];
	    argcIndex += 2;
	    break;
	  }
	case 'd':
	  {
	    if(argcIndex == argc - 1 || argv[argcIndex+1][0] == '-')
	      {
		cout << "no output directory provided" << endl;
		return -1;
	      }
	    display = true;
	    dDirectory = argv[argcIndex+1];
	    argcIndex += 2;
	    break;
	  }
	case 'i':
	  {
	    if(argcIndex >= argc - 2 || argv[argcIndex+1][0] == '-'
	       || argv[argcIndex+2][0] == '-')
	      {
		cout << "not enough parameters specified" << endl;
		return -1;
	      }
	    innerDimW = atoi(argv[argcIndex+1]);
	    innerDimH = atoi(argv[argcIndex+2]);
	    argcIndex += 3;
	    break;
	  }
	case 'l':
	  {
	    if(argcIndex == argc - 1 || argv[argcIndex+1][0] == '-')
	      {
		cout << "number of pixel locations not specified" << endl;
		return -1;
	      }
	    numRandLoc = atoi(argv[argcIndex+1]);
	    argcIndex += 2;
	    break;
	  }
	case 'c':
	  {
	    if(argcIndex == argc - 1 || argv[argcIndex+1][0] == '-')
	      {
		cout << "number of clusters not specified" << endl;
		return -1;
	      }
	    numFeatureClusters = atoi(argv[argcIndex+1]);
	    argcIndex += 2;
	    break;
	  }
	case 'n':
	  {
	    nonRand = true;
	    ++argcIndex;
	    break;
	  }
	default:
	  cout << "unrecognized option" << endl;
	  return -1;
	}
    }

  ifstream imageList(argv[1]);
  ofstream trainingInfo;

  if(!imageList.is_open())
    {
      cout << "couldn't open " << argv[1] << " for reading" << endl;
      return -1;
    }
  if(!animation)
    {
      trainingInfo.open(argv[2]);
      if(!trainingInfo.is_open())
	{
	  cout << "couldn't open " << argv[2] << " for writing" << endl;
	  return -1;
	}
    }

  if(outerDimW - innerDimW < 2*windowSize)
    {
      cout << "difference between outerDimW and innerDimW is not greater than window size for SIFT descriptor)" << endl;
      return -1;
    }
  if( (outerDimW - innerDimW) % 2 != 0)
    {
      cout << "shrinking innerDimW by 1 so outerDimW - innerDimW is divisible by 2" << endl;
      --innerDimW;
    }
  int paddingW = ((outerDimW - innerDimW) - 2*windowSize) / 2;

  if(outerDimH - innerDimH < 2*windowSize)
    {
      cout << "difference between outerDimH and innerDimH is not greater than window size for SIFT descriptor)" << endl;
      return -1;
    }
  if( (outerDimH - innerDimH) % 2 != 0)
    {
      cout << "shrinking innerDimH by 1 so outerDimH - innerDimH is divisible by 2" << endl;
      --innerDimH;
    }
  int paddingH = ((outerDimH - innerDimH) - 2*windowSize) / 2;
	
  // smoothing to avoid zero probabilities
  float smoothingParam = 0.1;
  float smoothingNormalize = smoothingParam * numFeatureClusters;

  // SIFT settings
  int siftHistDim = 4;
  int siftBucketsDim = 8;
  int siftDescDim = (4*windowSize*windowSize*siftBucketsDim)/(siftHistDim*siftHistDim);


  ofstream paramOutfile;
  if(outputParams)
    {
      paramOutfile.open(oaFilename.c_str());
      if(!paramOutfile.is_open())
	{
	  cout << "error opening " << oaFilename << " for writing" << endl;
	  return -1;
	}
    }
  

  // Section 1 //// ////////////////////////////////////////////////////////////////////////////////////////////////
  // preliminaries : read in images, do animation if necessary

  vector<IplImage *> baseImages;
  vector<IplImage *> originalImages;
  vector<vector<float> > cropParams;
  vector<int> imageIndex;
  string imageFn;

  while(true)
    {
      imageList >> imageFn;
      if(imageList.eof())
	break;

      IplImage *tmp;
      if((tmp = cvLoadImage(imageFn.c_str(), -1)) == 0)
	{
	  cout << "error loading image: " << imageFn << endl;
	  return -1;
	}

      // resize to outerDim
      IplImage *tmp2 = cvCreateImage(cvSize(outerDimW, outerDimH), tmp->depth, tmp->nChannels);
      cvResize(tmp, tmp2);
      // change to floating point
      IplImage *tmp3 = cvCreateImage(cvSize(outerDimW, outerDimH), IPL_DEPTH_32F, tmp->nChannels);
      cvConvertScale(tmp2, tmp3);
      // convert to grayscale
      if(tmp3->nChannels > 1)
	{
	  IplImage *tmp4 = cvCreateImage(cvSize(outerDimW, outerDimH), IPL_DEPTH_32F, 1);
	  cvCvtColor(tmp3, tmp4, CV_BGR2GRAY);
	  IplImage *tmp5 = tmp3;
	  tmp3 = tmp4;
	  cvReleaseImage(&tmp5);
	}

      baseImages.push_back(tmp3);
      if(animation)
	originalImages.push_back(tmp);
      else
	cvReleaseImage(&tmp);
      cvReleaseImage(&tmp2);
    }
  imageList.close();

  int numImages = baseImages.size();
  vector<float> vrow(numParams);
  vector<vector<float> > v(numImages, vrow);
  vector<vector<float> > vOld(numImages, vrow);
  float distFieldNormalize = numImages + smoothingNormalize;

  const int height = baseImages[0]->height-2*windowSize;
  const int width  = baseImages[0]->width-2*windowSize;
  const int baseWidthStep = baseImages[0]->widthStep / sizeof(float);

  CvMat *allFeatures = cvCreateMat(numImages*(height-2*paddingH)*(width-2*paddingW)/4, siftDescDim, CV_32FC1);
  int AFi = 0;
  vector<float> ofEntry(siftDescDim, 0);
  vector<vector<float> > ofCol(width, ofEntry);
  vector<vector<vector<float> > > ofRow(height, ofCol);
  vector<vector<vector<vector<float> > > > originalFeatures(numImages, ofRow);
  vector<float> SiftDesc(siftDescDim);
  
  // make animation and quit
  if(animation)
    {
      ifstream infile(oaFilename.c_str());
      if(!infile.is_open())
	{
	  cout << "couldn't open " << oaFilename << " for reading" << endl;
	  return -1;
	}

      vector<float> zr(numParams, 0);
      vector<vector<float> > z(numImages, zr);
      vector<vector<vector<float> > > iterParams(1, z);
      while(true)
	{
	  vector<float> p(numParams);
	  for(int j=0; j<numParams; j++)
	    infile >> p[j];
	  if(infile.eof())
	    break;
	  vector<vector<float> > iterp(1, p);
	  for(int i=1; i<numImages; i++)
	    {
	      for(int j=0; j<numParams; j++)
		infile >> p[j];
	      iterp.push_back(p);
	    }
	  iterParams.push_back(iterp);
	}
      
      // calculate original square from last params
      vector<vector<vector<float> > > bSq(numImages);
      for(int i=0; i<numImages; i++)
	{
	  int w = outerDimW, h = outerDimH;
	  vector<vector<float> > v = iterParams[iterParams.size()-1];
	  float postM[2][3] = {{1,0,w/2.0f}, {0,1,h/2.0f}};
	  float preM[3][3] = {{1,0,-w/2.0f}, {0,1,-h/2.0f}, {0,0,1}};

	  float tM[3][3]  = {{1, 0, v[i][0]}, {0, 1, v[i][1]}, {0,0,1}};
	  float rM[3][3]  = {{cos(v[i][2]), -sin(v[i][2]), 0}, {sin(v[i][2]), cos(v[i][2]), 0}, {0, 0, 1}};
	  float sM[3][3]  = {{exp(v[i][3]), 0, 0}, {0, exp(v[i][3]), 0}, {0, 0, 1}};
	  
	  CvMat tCVM, rCVM, sCVM, hxCVM, hyCVM, *xform, postCVM, preCVM;
	  tCVM  = cvMat(3, 3, CV_32FC1, tM);
	  rCVM  = cvMat(3, 3, CV_32FC1, rM);
	  sCVM  = cvMat(3, 3, CV_32FC1, sM);
	  
	  postCVM = cvMat(2, 3, CV_32FC1, postM);
	  preCVM = cvMat(3, 3, CV_32FC1, preM);
	  
	  xform = cvCreateMat(2, 3, CV_32FC1);
	  cvMatMul(&postCVM, &tCVM, xform);
	  cvMatMul(xform, &rCVM, xform);
	  cvMatMul(xform, &sCVM, xform);
	  cvMatMul(xform, &preCVM, xform);
	  
	  int sqPts[4][2] = {{paddingW, paddingH}, {w-paddingW, paddingH}, 
			     {w-paddingW, h-paddingH}, {paddingW, h-paddingH}};
	  vector<float> biter2(2);
	  vector<vector<float> > biter(5, biter2);
	  for(int b=0; b<4; b++)
	    {
	      biter[b][0] = xform->data.fl[0]*sqPts[b][0] + xform->data.fl[1]*sqPts[b][1] + xform->data.fl[2];
	      biter[b][1] = xform->data.fl[3]*sqPts[b][0] + xform->data.fl[4]*sqPts[b][1] + xform->data.fl[5];
	    }	    
	  biter[4][0] = biter[0][0]; biter[4][1] = biter[0][1];
	  bSq[i] = biter;
	}
      
      // for each iteration, create picture
      for(int i=0; i<iterParams.size(); i++)
	MakeFrame(originalImages, iterParams[i], outerDimH, outerDimW, bSq, aDirectory, i, maxFrameIndex);

      for(int i=0; i<numImages; i++)
	cvReleaseImage(&originalImages[i]);

      // entropy animation
      ifstream modelFile(argv[2]);
      int edgeDescDim;
      modelFile >> numFeatureClusters >> edgeDescDim;
      vector<float> cRow(edgeDescDim, 0);
      vector<vector<float> > centroids(numFeatureClusters, cRow);
      vector<float> sigmaSq(numFeatureClusters);
      
      for(int i=0; i<numFeatureClusters; i++)
	{
	  for(int j=0; j<edgeDescDim; j++)
	    {
	      modelFile >> centroids[i][j];
	    }
	  modelFile >> sigmaSq[i];
	}
      modelFile >> numRandLoc;
      vector<pair<int, int> > randLocs(numRandLoc);
      for(int j=0; j<numRandLoc; j++)
	modelFile >> randLocs[j].first >> randLocs[j].second;

      vector<float> dfRow(numFeatureClusters);
      vector<vector<float> > logDF(numRandLoc, dfRow);
      vector<vector<vector<float> > > logDFSeq;
      int iter;

      float minEnt = FLT_MAX, maxEnt = 0;

      while(true)
	{
	  modelFile >> iter;
	  if(modelFile.eof())
	    break;
	  for(int j=0; j<numRandLoc; j++)
	    {
	      float h = 0;
	      for(int i=0; i<numFeatureClusters; i++)
		{
		  modelFile >> logDF[j][i];
		  h -= logDF[j][i] * exp(logDF[j][i]);
		}
	      if(h > maxEnt)
		maxEnt = h;
	      if(h < minEnt)
		minEnt = h;
	    }
	  logDFSeq.push_back(logDF);
	}

      char efn[1024];
      for(int i=0; i<logDFSeq.size(); i++)
	{
	  sprintf(efn, "%s/DFEnt_%03d.jpg", aDirectory.c_str(), i);
	  makeEntFrame(logDFSeq[i], randLocs, efn, minEnt, maxEnt, innerDimH, innerDimW, paddingH, paddingW);
	}

      return 0;
    }


  // Section 2 ///////////////////////////////////////////////////////////////////////////////////////////////////////
  // compute SIFT descriptors for window around each pixel, run kmeans, assign mixture of centroids to each pixel

  vector<float> mtRow(width+2*windowSize);
  vector<vector<float> > m(height+2*windowSize, mtRow);
  vector<vector<float> > theta(height+2*windowSize, mtRow);
  float dx, dy;
  vector<vector<float> > Gaussian;
  computeGaussian(Gaussian, windowSize);

  for(int i=0; i<numImages; i++)
    {
      for(int j=0; j<height+2*windowSize; j++)
	{
	  for(int k=0; k<width+2*windowSize; k++)
	    {
	      if(j==0)
		dy = ((float*)baseImages[i]->imageData)[(j+1)*baseWidthStep+k] - ((float*)baseImages[i]->imageData)[j*baseWidthStep+k];
	      else
		{
		  if(j==height+2*windowSize-1)
		    dy = ((float*)baseImages[i]->imageData)[j*baseWidthStep+k] - ((float*)baseImages[i]->imageData)[(j-1)*baseWidthStep+k];
		  else
		    dy = ((float*)baseImages[i]->imageData)[(j+1)*baseWidthStep+k] - ((float*)baseImages[i]->imageData)[(j-1)*baseWidthStep+k];
		}
	      if(k==0)
		dx = ((float*)baseImages[i]->imageData)[j*baseWidthStep+(k+1)] - ((float*)baseImages[i]->imageData)[j*baseWidthStep+k];
	      else
		{
		  if(k==width+2*windowSize-1)
		    dx = ((float*)baseImages[i]->imageData)[j*baseWidthStep+k] - ((float*)baseImages[i]->imageData)[j*baseWidthStep+(k-1)];
		  else
		    dx = ((float*)baseImages[i]->imageData)[j*baseWidthStep+(k+1)] - ((float*)baseImages[i]->imageData)[j*baseWidthStep+(k-1)];
		}
	      
	      m[j][k] = (float)sqrt(dx*dx+dy*dy);
	      theta[j][k] = (float)atan2(dy,dx) * 180.0f/pi;
	      if(theta[j][k] < 0)
		theta[j][k] += 360.0f;
	    }
	}

      for(int j=0; j<height; j++)
	{
	  for(int k=0; k<width; k++)
	    {
	      getSIFTdescripter(SiftDesc, m, theta, j+windowSize, k+windowSize, windowSize, siftHistDim, siftBucketsDim, Gaussian);
	      
	      if(j >= paddingH && j < height-paddingH && k >= paddingW && k < width-paddingW && (j-paddingH)%2==0 && (k-paddingW)%2==0)
		{
		  for(int ii=0; ii<(signed)SiftDesc.size(); ii++)
		    {
		      allFeatures->data.fl[AFi*siftDescDim + ii] = SiftDesc[ii];
		    }
		  ++AFi;
		}
	      originalFeatures[i][j][k] = SiftDesc;
	    }
	}
    }

  // kmeans
  CvMat *labels = cvCreateMat(allFeatures->height, 1, CV_32SC1);
  cvKMeans2(allFeatures, numFeatureClusters, labels, 
	    cvTermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 1000, .001));
  
  vector<float> cRow(siftDescDim, 0);
  vector<vector<float> > centroids(numFeatureClusters, cRow);
  
  if(nonRand)
    numRandLoc = (height-2*paddingH) * (width-2*paddingW);
  vector<pair<int, int> > randLocs(numRandLoc);
  setRandLocs(randLocs, height-2*paddingH, width-2*paddingW, paddingH, paddingW, nonRand);
  
  vector<float> fidsEntry(numFeatureClusters, 0);
  vector<vector<float> > fidsRow(numRandLoc, fidsEntry);
  vector<vector<vector<float> > > featureIDs(numImages, fidsRow);
  vector<float> dfCol(numFeatureClusters, 0);
  vector<vector<float> > distField(numRandLoc, dfCol);
  vector<int> numInC(numFeatureClusters, 0);
  vector<float> sigmaSq(numFeatureClusters);
  
  // compute centroids and sigmaSq
  AFi = 0;
  for(int i=0; i<numImages; i++)
    {
      for(int j=paddingH; j<height-paddingH; j+=2)
	{
	  for(int k=paddingW; k<width-paddingW; k+=2)
	    {
	      int label = labels->data.i[AFi];
	      
	      for(int ii=0; ii<siftDescDim; ii++)
		centroids[label][ii] += allFeatures->data.fl[AFi*siftDescDim+ii];
	      
	      ++numInC[label];
	      ++AFi;
	    }
	}
    }
  for(int i=0; i<numFeatureClusters; i++)
    {
      for(int j=0; j<siftDescDim; j++)
	{
	  centroids[i][j] /= numInC[i];
	}
    }
  
  AFi = 0;
  for(int i=0; i<numImages; i++)
    {
      for(int j=paddingH; j<height-paddingH; j+=2)
	{
	  for(int k=paddingW; k<width-paddingW; k+=2)
	    {
	      int label = labels->data.i[AFi];
	      sigmaSq[label] += dist(originalFeatures[i][j][k], centroids[label]);
	      ++AFi;
	    }
	}
    }
  for(int i=0; i<numFeatureClusters; i++)
    {
      sigmaSq[i] /= numInC[i];
      if(sigmaSq[i] < .000001)
	sigmaSq[i] = .000001;
    }
  
  cvReleaseMat(&allFeatures);
  cvReleaseMat(&labels);
  
  trainingInfo << numFeatureClusters << " " << siftDescDim << endl;
  for(int i=0; i<numFeatureClusters; i++)
    {
      for(int j=0; j<siftDescDim; j++)
	{
	  trainingInfo << centroids[i][j] << " ";
	}
      trainingInfo << sigmaSq[i] << endl;
    }
  trainingInfo << endl;
  
  // compute mixture of centroids
  // visualization information
  vector<int> highestMatchesRow(3, -1);
  vector<vector<int> > highestMatchesCol(numFeatureClusters, highestMatchesRow);
  vector<vector<vector<int> > > highestMatches(numFeatureClusters, highestMatchesCol);
  
  vector<float> highestSiftRow(numFeatureClusters, 0);
  vector<vector<float> > highestSiftCol(numFeatureClusters, highestSiftRow);
  vector<vector<vector<float> > > highestSift(numFeatureClusters, highestSiftCol);
  
  vector<float> highestValsRow(numFeatureClusters, -1);
  vector<vector<float> > highestVals(numFeatureClusters, highestValsRow);
  
  for(int i=0; i<numImages; i++)
    {
      for(int j=0; j<height; j++)
	{
	  for(int k=0; k<width; k++)
	    {
	      vector<float> distances(numFeatureClusters);
	      float sum = 0, sum2 = 0;
	      for(int ii=0; ii<numFeatureClusters; ii++)
		{
		  if(sigmaSq[ii] > 0)
		    distances[ii] = exp(-dist(originalFeatures[i][j][k], centroids[ii])/(2*sigmaSq[ii]))/sqrt(sigmaSq[ii]);
		  sum += distances[ii];
		}
	      for(int ii=0; ii<numFeatureClusters; ii++)
		distances[ii] /= sum;
	      
	      for(int f=0; f<numFeatureClusters; f++)
		{
		  for(int index=0; index<numFeatureClusters; index++)
		    {
		      if(distances[f] > highestVals[f][index]) 
			{
			  vector<int> newMatch(3);
			  newMatch[0] = i;
			  newMatch[1] = j;
			  newMatch[2] = k;
			  highestMatches[f].insert(highestMatches[f].begin()+index, newMatch);
			  highestMatches[f].erase(highestMatches[f].begin()+numFeatureClusters);
			  highestVals[f].insert(highestVals[f].begin()+index, distances[f]);
			  highestVals[f].erase(highestVals[f].begin()+numFeatureClusters);
			  highestSift[f].insert(highestSift[f].begin()+index, originalFeatures[i][j][k]);
			  highestSift[f].erase(highestSift[f].begin()+numFeatureClusters);
			  break;
			}
		    }
		}	      
	      
	      originalFeatures[i][j][k] = distances;
	    }
	}
    }
  
  if(visualize)
    {
      int patchDimP1 = 4*windowSize+1;
      IplImage *vCImage = cvCreateImage(cvSize(numFeatureClusters*patchDimP1+1, numFeatureClusters*patchDimP1+1), 
					IPL_DEPTH_32F, 1);
      IplImage *patchImage = cvCreateImage(cvSize(2*windowSize, 2*windowSize), IPL_DEPTH_32F, 1);
      for(int i=0; i<numFeatureClusters; i++)
	{
	  for(int j=0; j<numFeatureClusters; j++)
	    {
	      cvSetImageROI(baseImages[highestMatches[i][j][0]], 
			    cvRect(highestMatches[i][j][2] - (windowSize-1),
				   highestMatches[i][j][1] - (windowSize-1), 2*windowSize, 2*windowSize));
	      cvResize(baseImages[highestMatches[i][j][0]], patchImage);
	      cvResetImageROI(baseImages[highestMatches[i][j][0]]);
	      
	      cvSetImageROI(vCImage, cvRect(j*patchDimP1+1, i*patchDimP1+1, 4*windowSize, 4*windowSize));
	      cvResize(patchImage, vCImage);
	      cvResetImageROI(vCImage);
	    }
	}

      IplImage *vc8U = cvCreateImage(cvSize(vCImage->width, vCImage->height), IPL_DEPTH_8U, 1);
      IplImage *vc8U3 = cvCreateImage(cvSize(vCImage->width, vCImage->height), IPL_DEPTH_8U, 3);
      
      cvConvertScale(vCImage, vc8U, 1);
      cvCvtColor(vc8U, vc8U3, CV_GRAY2BGR);
      
      cvNamedWindow("CR", 1);
      cvShowImage("CR", vc8U3);
      cvWaitKey();
      cvDestroyWindow("CR");
      
      char s[1024];
      sprintf(s, "%s/clusterReps.png", vDirectory.c_str());
      cvSaveImage(s, vc8U3);
      
      cvReleaseImage(&vc8U);
      cvReleaseImage(&vc8U3);
      
      cvReleaseImage(&patchImage);
      cvReleaseImage(&vCImage);
    }
  
  trainingInfo << numRandLoc << endl;
  for(int j=0; j<numRandLoc; j++)
    trainingInfo << randLocs[j].first << " " << randLocs[j].second << endl;
  trainingInfo << endl;
  
  // set initial random pixel mixture of centroids
  for(int i=0; i<numImages; i++)
    {
      for(int j=0; j<numRandLoc; j++)
	{
	  featureIDs[i][j] = originalFeatures[i][randLocs[j].first][randLocs[j].second];
	}
    }
  
  // compute initial distribution field
  trainingInfo << 0 << endl;
  float obj = 0;
  for(int j=0; j<numRandLoc; j++)
    {
      for(int i=0; i<numFeatureClusters; i++)
	distField[j][i] = smoothingParam / distFieldNormalize;
      for(int i=0; i<numImages; i++)
	{
	  for(int ii=0; ii<numFeatureClusters; ii++)
	    distField[j][ii] += (featureIDs[i][j][ii] / distFieldNormalize);
	}
      for(int i=0; i<numFeatureClusters; i++)
	trainingInfo << log(distField[j][i]) << " ";
      trainingInfo << endl;
    }
  trainingInfo << endl;
  
  float d[numParams] = {1.0f, 1.0f, pi/180.0f, 0.02f};
  
  vector<float> nfEntry(numFeatureClusters, 0);
  vector<vector<float> > newFIDs(numRandLoc, nfEntry);
  float centerX = width/2.0f, centerY = height/2.0f;

  for(int i=0; i<numImages; i++)
    cvReleaseImage(&baseImages[i]);  

  // Section 3 ///////////////////////////////////////////////////////////////////////////////////////////////////////////
  // main congealing algorithm
  
  int iter=0;
  float oldEntropy = FLT_MAX;
  float newEntropy = FLT_MAX/2.0f;
  while(oldEntropy - newEntropy > 0.5f)
    {
      oldEntropy = newEntropy;
      ++iter;

      if(verbose) cout << "iteration " << iter << ", ";

      if(iter > maxIters)
	{
	  cout << "didn't converge" << endl;
	  return 0;
	}
      
      for(int j=0; j<numImages; j++)
	{
	  float oldL = computeLogLikelihood(distField, featureIDs[j], numFeatureClusters);
	  
	  for(int k=0; k<numParams; k++)
	    {	
	      float dn = ((rand()%160)-80)/100.0f;
	      if(k>1)
		dn /= 100.0f;
	      v[j][k] += (d[k] + dn);
	      
	      getNewFeatsInvT(newFIDs, originalFeatures[j], v[j], centerX, centerY, randLocs);
	      float newL = computeLogLikelihood(distField, newFIDs, numFeatureClusters);
	      
	      if(newL > oldL)
		{
		  featureIDs[j] = newFIDs;
		  oldL = newL;
		}
	      else
		{
		  v[j][k] -= (2*(d[k] + dn));
		  getNewFeatsInvT(newFIDs, originalFeatures[j], v[j], centerX, centerY, randLocs);
		  newL = computeLogLikelihood(distField, newFIDs, numFeatureClusters);
		  
		  if(newL > oldL)
		    {
		      oldL = newL;
		      featureIDs[j] = newFIDs;
		    }
		  else
		    v[j][k] += (d[k]+dn);
		}
	    }
	}

      // balancing scale
      vector<float> vbar(numParams,0);
      for(int k=numParams-1; k<numParams; k++)
	{
	  for(int j=0; j<numImages; j++)
	    vbar[k] += v[j][k];
	  vbar[k] /= numImages;
	}

      for(int j=0; j<numImages; j++)
	{
	  for(int k=numParams-1; k<numParams; k++)
	    v[j][k] -= vbar[k];
	}
      
      for(int j=0; j<numImages; j++)
	getNewFeatsInvT(featureIDs[j], originalFeatures[j], v[j], centerX, centerY, randLocs);
      
      /*
      // code for resampling, not used
      setRandLocs(randLocs, height-2*padding, width-2*padding, padding);	
      for(int i=0; i<numImages; i++)
	getNewFeatsInvT(featureIDs[i], originalFeatures[i], padding, v[i], centerX, centerY, randLocs, flag);
      */
      
      // recompute and write out distribution field
      trainingInfo << iter << endl;

      for(int j=0; j<numRandLoc; j++)
	{
	  for(int i=0; i<numFeatureClusters; i++)
	    distField[j][i] = smoothingParam / distFieldNormalize;
	  for(int i=0; i<numImages; i++)
	    {
	      for(int ii=0; ii<numFeatureClusters; ii++)
		distField[j][ii] += (featureIDs[i][j][ii] / distFieldNormalize);
	    }
	  for(int i=0; i<numFeatureClusters; i++)
	    trainingInfo << log(distField[j][i]) << " ";
	  trainingInfo << endl;
	}
      trainingInfo << endl;

      newEntropy = computeEntropy(distField, numFeatureClusters);
      if(verbose) cout << "entropy: " << newEntropy << endl;

      if(outputParams)
	{
	  for(int j=0; j<numImages; j++)
	    {
	      for(int k=0; k<numParams; k++)
		paramOutfile << v[j][k] << " ";
	      paramOutfile << endl;
	    }
	}
    }
  
  // Section 4 ///////////////////////////////////////////////////////////////////////////////////////////////////////
  // display results

  if(display || generateFinal)
    {
      vector<pair<int, float> > indexProbPairs;
      for(int j=0; j<numImages; j++)
	{
	  pair<int, float> p(j, computeLogLikelihood(distField, featureIDs[j], numFeatureClusters));
	  indexProbPairs.push_back(p);
	}
      //sort(indexProbPairs.begin(), indexProbPairs.end(), indexProbComparison);
      
      showResults(argv[1], v, outerDimH, outerDimW, paddingH, paddingW, indexProbPairs, display, dDirectory, generateFinal, gDirectory);
    }

  if(visualize)
    {
      IplImage *finalDFEnt = cvCreateImage(cvSize(innerDimW, innerDimH), IPL_DEPTH_8U, 3);
      cvSetZero(finalDFEnt);
      int entStep = finalDFEnt->widthStep/sizeof(uchar);
      uchar *entData = (uchar *)finalDFEnt->imageData;

      vector<float> entropies(numRandLoc, 0);
      float minEnt = 100, maxEnt = 0;
      for(int i=0; i<randLocs.size(); i++)
	{	  
	  for(int j=0; j<numFeatureClusters; j++)
	    entropies[i] -= distField[i][j] * log2(distField[i][j]);
	  if(entropies[i] < minEnt)
	    minEnt = entropies[i];
	  if(entropies[i] > maxEnt)
	    maxEnt = entropies[i];
	}
      float entSpread = maxEnt - minEnt;

      for(int i=0; i<randLocs.size(); i++)
	{
	  int j = randLocs[i].first - paddingH, k = randLocs[i].second - paddingW;
	  int val = (int)( 255.0*(entropies[i]-minEnt) / entSpread);
	  entData[j*entStep+k*3] = 255 - val;
	  entData[j*entStep+k*3+2] = val;
	}
      string outputFn = vDirectory + "/finalDFEnt.jpg";
      cvNamedWindow("Final DF Ent", 1);
      cvShowImage("Final DF Ent", finalDFEnt);
      cvWaitKey();
      cvDestroyWindow("Final DF Ent");
      cvSaveImage(outputFn.c_str(), finalDFEnt);
      cvReleaseImage(&finalDFEnt);
    }
  
  return 0;
}

void makeEntFrame(vector<vector<float> > &logDistField, vector<pair<int, int> > &randLocs, char *fn, float minEnt, float maxEnt, 
		  int innerDimH, int innerDimW, int paddingH, int paddingW)
{
  IplImage *DFEnt = cvCreateImage(cvSize(innerDimW, innerDimH), IPL_DEPTH_8U, 3);
  cvSetZero(DFEnt);
  int entStep = DFEnt->widthStep/sizeof(uchar);
  uchar *entData = (uchar *)DFEnt->imageData;
  
  vector<float> entropies(randLocs.size(), 0);

  for(int i=0; i<randLocs.size(); i++)
    {	  
      for(int j=0; j<logDistField[i].size(); j++)
	entropies[i] -= logDistField[i][j] * exp(logDistField[i][j]);
    }
  float entSpread = maxEnt - minEnt;
  
  for(int i=0; i<randLocs.size(); i++)
    {
      int j = randLocs[i].first - paddingH, k = randLocs[i].second - paddingW;
      int val = (int)( 255.0*(entropies[i]-minEnt) / entSpread);
      entData[j*entStep+k*3] = 255 - val;
      entData[j*entStep+k*3+2] = val;
    }
  cvSaveImage(fn, DFEnt);
  cvReleaseImage(&DFEnt);
}

void MakeFrame(vector<IplImage *> &images, vector<vector<float> > &v, int h, int w,
	       vector<vector<vector<float> > > &bSq, string basefn, int frameIndex, int maxFrameIndex)
{
  char s[1024];

  for(int index=1; index<= ((signed)images.size() + 24) / 25; index++)
    {
      if(index > maxFrameIndex)
	break;

      int maxdim = (h > w) ? h : w;
      float scale = maxdim/90.0f;
      
      int minbound = ((signed)images.size() > 25*index) ? 25 : (int)images.size()-25*(index-1);

      IplImage *resultsT = cvCreateImage(cvSize(500, 500), images[0]->depth, 3);
      for(int i=0; i<minbound; i++)
	{
	  int xs = i % 5;
	  int ys = i / 5;

	  cvSetImageROI(resultsT, cvRect(xs*100+1, ys*100+1, (int)(w/scale), (int)(h/scale)));
	  
	  int ii=i+25*(index-1);

	  float cropT1inv[2][3] = {{1,0,images[ii]->width/2.0f}, {0,1,images[ii]->height/2.0f}};
	  float cropS1inv[3][3] = {{images[ii]->width/(float)w,0,0}, {0,images[ii]->height/(float)h,0}, {0,0,1}};
	  float cropS2inv[3][3] = {{w/(float)images[ii]->width,0,0}, {0,h/(float)images[ii]->height,0}, {0,0,1}};		    
	  float cropT2inv[3][3] = {{1,0,-images[ii]->width/2.0f}, {0,1,-images[ii]->height/2.0f}, {0,0,1}};

	  float postM[3][3] = {{1,0,w/2.0f}, {0,1,h/2.0f}, {0,0,1}};
	  float preM[3][3] = {{1,0,-w/2.0f}, {0,1,-h/2.0f}, {0,0,1}};
	  
	  float tM[3][3]  = {{1, 0, v[ii][0]}, {0, 1, v[ii][1]}, {0,0,1}};
	  float rM[3][3]  = {{cos(v[ii][2]), -sin(v[ii][2]), 0}, {sin(v[ii][2]), cos(v[ii][2]), 0}, {0, 0, 1}};
	  float sM[3][3]  = {{exp(v[ii][3]), 0, 0}, {0, exp(v[ii][3]), 0}, {0, 0, 1}};
	  
	  CvMat tCVM, rCVM, sCVM, hxCVM, hyCVM, *xform, postCVM, preCVM, cropT1invCVM, cropS1invCVM, cropT2invCVM, cropS2invCVM;
	  
	  tCVM  = cvMat(3, 3, CV_32FC1, tM);
	  rCVM  = cvMat(3, 3, CV_32FC1, rM);
	  sCVM  = cvMat(3, 3, CV_32FC1, sM);
	  
	  postCVM = cvMat(3, 3, CV_32FC1, postM);
	  preCVM = cvMat(3, 3, CV_32FC1, preM);
	  cropT1invCVM = cvMat(2,3,CV_32FC1, cropT1inv);
	  cropS1invCVM = cvMat(3,3,CV_32FC1, cropS1inv);
	  cropS2invCVM = cvMat(3,3,CV_32FC1, cropS2inv);
	  cropT2invCVM = cvMat(3,3,CV_32FC1, cropT2inv);
	  
	  IplImage *dst;

	  xform = cvCreateMat(2, 3, CV_32FC1);
	  cvMatMul(&cropT1invCVM, &cropS1invCVM, xform);
	  cvMatMul(xform, &tCVM, xform);
	  cvMatMul(xform, &rCVM, xform);
	  cvMatMul(xform, &sCVM, xform);
	  cvMatMul(xform, &preCVM, xform);
	  
	  dst =  cvCreateImage(cvSize(w, h), images[ii]->depth, images[ii]->nChannels);
	  
	  CvMat *xform2 = cvCreateMat(3,3,CV_32FC1), *xformInv = cvCreateMat(3,3,CV_32FC1);
	  cvMatMul(&postCVM, &tCVM, xform2);
	  cvMatMul(xform2, &rCVM, xform2);
	  cvMatMul(xform2, &sCVM, xform2);
	  cvMatMul(xform2, &preCVM, xform2);
	  cvInvert(xform2, xformInv);
	  
	  cvWarpAffine(images[ii], dst, xform, CV_WARP_INVERSE_MAP + CV_WARP_FILL_OUTLIERS + CV_INTER_LINEAR);
	  
	  int BsqPts[5][2];
	  for(int b=0; b<5; b++)
	    {
	      BsqPts[b][0] = (int)(xformInv->data.fl[0]*bSq[ii][b][0] + xformInv->data.fl[1]*bSq[ii][b][1] + xformInv->data.fl[2] + 0.5f);
	      BsqPts[b][1] = (int)(xformInv->data.fl[3]*bSq[ii][b][0] + xformInv->data.fl[4]*bSq[ii][b][1] + xformInv->data.fl[5] + 0.5f);
	    }
	  for(int b=0; b<4; b++)
	    cvLine(dst, cvPoint(BsqPts[b][0], BsqPts[b][1]), cvPoint(BsqPts[b+1][0], BsqPts[b+1][1]), cvScalar(0,0,255),2);
	  
	  cvReleaseMat(&xform);
	  
	  cvResize(dst, resultsT);
	  cvReleaseImage(&dst); 

	  cvResetImageROI(resultsT);
	  cvRectangle(resultsT, cvPoint(xs*100, ys*100), 
		      cvPoint(xs*100 + (int)(w/scale) + 1, ys*100 + (int)(h/scale) + 1), cvScalar(255,0,0), 1);
	}
      sprintf(s, "%s/animation_%02d_%03d.png", basefn.c_str(), index, frameIndex);
      cvSaveImage(s, resultsT);
      cvReleaseImage(&resultsT);
    } 
}

void showResults(char *imageListFn, vector<vector<float> > &v, int h, int w,
		 int paddingH, int paddingW, vector<pair<int,float> > &indexProbPairs,
		 bool display, string dDirectory, bool generateFinal, string gDirectory)
{
  bool skipFlag = false;
  string cmd = "test -d " + gDirectory;
  bool isDir = !system(cmd.c_str());
  ifstream ofnsfile;
  if(!isDir)
    {
      ofnsfile.open(gDirectory.c_str());
      if(!ofnsfile.is_open())
	{
	  cout << "could not open " << gDirectory << " for reading" << endl;
	  exit(-1);
	}
    }

  if(display)
    {
      cvNamedWindow("Results", 1);
      cvNamedWindow("ResultsT", 1);
    }

  vector<IplImage *> images;
  vector<string> imageFns;
  string imageFn;
  ifstream imageList(imageListFn);

  while(true)
    {
      imageList >> imageFn;
      if(imageList.eof())
	break;
      
      imageFns.push_back(imageFn);
      IplImage *tmp;
      if((tmp = cvLoadImage(imageFn.c_str(), -1)) == 0)
	{
	  cout << "error loading image: " << imageFn << endl;
	  exit(-1);
	}
      images.push_back(tmp);
    }
	  
  IplImage *results = cvCreateImage(cvSize(500, 500), images[0]->depth, 3);
  IplImage *resultsT = cvCreateImage(cvSize(500, 500), images[0]->depth, 3);
  cvSetZero(results);
  cvSetZero(resultsT);
  
  for(int index=1; index<= ((signed)images.size() + 24) / 25; index++)
    {
      int maxdim = (h > w) ? h : w;
      float scale = maxdim/90.0f;

      int minbound = ((signed)images.size() > 25*index) ? 25 : (int)images.size()-25*(index-1);

      for(int i=0; i<minbound; i++)
	{
	  int xs = i % 5;
	  int ys = i / 5;

	  cvSetImageROI(results, cvRect(xs*100+1, ys*100+1, (int)(w/scale), (int)(h/scale)));
	  cvSetImageROI(resultsT, cvRect(xs*100+1, ys*100+1, (int)(w/scale), (int)(h/scale)));

	  int ii=i+25*(index-1);
	  if(indexProbPairs.size() > 0)
	    ii = indexProbPairs[ii].first;
	  
	  IplImage *dst = cvCreateImage(cvSize(w, h), images[ii]->depth, 3);
	  
	  // initialize transformation matrices
	  float cropT1inv[2][3] = {{1,0,images[ii]->width/2.0f}, {0,1,images[ii]->height/2.0f}};
	  float cropS1inv[3][3] = {{images[ii]->width/(float)w,0,0}, {0,images[ii]->height/(float)h,0}, {0,0,1}};		    
	  float cropS2inv[3][3] = {{w/(float)images[ii]->width,0,0}, {0,h/(float)images[ii]->height,0}, {0,0,1}};		    
	  float cropT2inv[3][3] = {{1,0,-images[ii]->width/2.0f}, {0,1,-images[ii]->height/2.0f}, {0,0,1}};

	  float postM[2][3] = {{1,0,w/2.0f}, {0,1,h/2.0f}};
	  float preM[3][3] = {{1,0,-w/2.0f}, {0,1,-h/2.0f}, {0,0,1}};

	  float tM[3][3]  = {{1, 0, v[ii][0]}, {0, 1, v[ii][1]}, {0,0,1}};
	  float rM[3][3]  = {{cos(v[ii][2]), -sin(v[ii][2]), 0}, {sin(v[ii][2]), cos(v[ii][2]), 0}, {0, 0, 1}};
	  float sM[3][3]  = {{exp(v[ii][3]), 0, 0}, {0, exp(v[ii][3]), 0}, {0, 0, 1}};

	  CvMat tCVM, rCVM, sCVM, *xform, *xform2, postCVM, preCVM, cropT1invCVM, cropS1invCVM, cropS2invCVM, cropT2invCVM;	  

	  cropT1invCVM = cvMat(2,3,CV_32FC1, cropT1inv);
	  cropS1invCVM = cvMat(3,3,CV_32FC1, cropS1inv);
	  cropS2invCVM = cvMat(3,3,CV_32FC1, cropS2inv);
	  cropT2invCVM = cvMat(3,3,CV_32FC1, cropT2inv);
	  
	  postCVM = cvMat(2, 3, CV_32FC1, postM);
	  preCVM = cvMat(3, 3, CV_32FC1, preM);
	  
	  tCVM  = cvMat(3, 3, CV_32FC1, tM);
	  rCVM  = cvMat(3, 3, CV_32FC1, rM);
	  sCVM  = cvMat(3, 3, CV_32FC1, sM);
		    
	  // do before image with bounding box
	  if(display)
	    {
	      xform2 = cvCreateMat(2, 3, CV_32FC1);
	      cvMatMul(&cropT1invCVM, &cropS1invCVM, xform2);
	      cvMatMul(xform2, &preCVM, xform2);
	      
	      cvWarpAffine(images[ii], dst, xform2, CV_WARP_INVERSE_MAP + CV_WARP_FILL_OUTLIERS + CV_INTER_LINEAR);
	      
	      cvReleaseMat(&xform2);
	  
	      xform = cvCreateMat(2, 3, CV_32FC1);
	      cvMatMul(&postCVM, &tCVM, xform);
	      cvMatMul(xform, &rCVM, xform);
	      cvMatMul(xform, &sCVM, xform);
	      cvMatMul(xform, &preCVM, xform);
	      
	      int sqPts[4][2] = {{paddingW, paddingH}, {w-paddingW, paddingH}, 
				 {w-paddingW, h-paddingH}, {paddingW, h-paddingH}};
	      int BsqPts[5][2];
	      for(int b=0; b<4; b++)
		{
		  BsqPts[b][0] = (int)(xform->data.fl[0]*sqPts[b][0] + xform->data.fl[1]*sqPts[b][1] + xform->data.fl[2] + 0.5f);
		  BsqPts[b][1] = (int)(xform->data.fl[3]*sqPts[b][0] + xform->data.fl[4]*sqPts[b][1] + xform->data.fl[5] + 0.5f);
		}
	      BsqPts[4][0] = BsqPts[0][0]; BsqPts[4][1] = BsqPts[0][1];
	      for(int b=0; b<4; b++)
		cvLine(dst, cvPoint(BsqPts[b][0], BsqPts[b][1]), cvPoint(BsqPts[b+1][0], BsqPts[b+1][1]), cvScalar(0,0,255),2);
	      
	      cvResize(dst, results);

	      cvReleaseMat(&xform);
	    }
	  cvResetImageROI(results);
	  cvRectangle(results, cvPoint(xs*100, ys*100), 
		      cvPoint(xs*100 + (int)(w/scale) + 1, ys*100 + (int)(h/scale) + 1), cvScalar(255,0,0), 1);
	  
	  cvReleaseImage(&dst);
	  
	  // do after image
	  xform = cvCreateMat(2, 3, CV_32FC1);
	  cvMatMul(&cropT1invCVM, &cropS1invCVM, xform);
	  cvMatMul(xform, &tCVM, xform);
	  cvMatMul(xform, &rCVM, xform);
	  cvMatMul(xform, &sCVM, xform);
	  cvMatMul(xform, &cropS2invCVM, xform);
	  cvMatMul(xform, &cropT2invCVM, xform);
	  
	  dst =  cvCreateImage(cvSize(images[ii]->width, images[ii]->height), images[ii]->depth, images[ii]->nChannels);
	  cvWarpAffine(images[ii], dst, xform, CV_WARP_INVERSE_MAP + CV_WARP_FILL_OUTLIERS + CV_INTER_LINEAR);
	  
	  cvReleaseMat(&xform);
	  
	  if(display)
	    cvResize(dst, resultsT);

	  cvResetImageROI(resultsT);
	  cvRectangle(resultsT, cvPoint(xs*100, ys*100), 
		      cvPoint(xs*100 + (int)(w/scale) + 1, ys*100 + (int)(h/scale) + 1), cvScalar(255,0,0), 1);

	  if(generateFinal)
	    {
	      string outputFn;
	      if(isDir)		
		outputFn = gDirectory + "/" + imageFns[ii];
	      else
		ofnsfile >> outputFn;
	      cvSaveImage(outputFn.c_str(), dst);
	    }
	  
	  cvReleaseImage(&dst);	  
	}
      
      if(display)
	{
	  if(!skipFlag)
	    {
	      cvShowImage("Results", results);	  
	      cvShowImage("ResultsT", resultsT);
	      
	      if( (cvWaitKey() & 0xffff) == 27)
		skipFlag = true;
	    }
	  
	  char outputFn[1024];
	  sprintf(outputFn, "%s/before_%02d.png", dDirectory.c_str(), index);
	  cvSaveImage(outputFn, results);
	  sprintf(outputFn, "%s/after_%02d.png", dDirectory.c_str(), index);
	  cvSaveImage(outputFn, resultsT);
	}
    }

  cvDestroyWindow("Results");
  cvDestroyWindow("ResultsT");
    
  cvReleaseImage(&results);
  cvReleaseImage(&resultsT);
  
  for(int i=0; i<images.size(); i++)
    cvReleaseImage(&images[i]);
}

float computeLogLikelihood(vector<vector<float> > &distField, vector<vector<float> > &fids, int numFeatureClusters)
{
  float l = 0;
  for(int j=0; j<(signed)fids.size(); j++)
    {
      for(int i=0; i<numFeatureClusters; i++)
	l += fids[j][i] * (float)log(distField[j][i]);
    }
  return l;
}

float computeEntropy(vector<vector<float> > &distField, int numFeatureClusters)
{
  float h = 0;
  for(int j=0; j<distField.size(); j++)
    {
      float hh = 0;
      for(int i=0; i<numFeatureClusters; i++)
	hh -= distField[j][i] * log2(distField[j][i]);
      h += hh;
    }
  return h;
}

void getNewFeatsInvT(vector<vector<float> > &newFIDs, vector<vector<vector<float> > > &originalFeats, 
		     vector<float> &vparams, float centerX, float centerY, vector<pair<int, int> > &randLocs)
{
  int numFeats = newFIDs[0].size();
  vector<float> uniformDist(numFeats, 1.0f/numFeats);

  float postM[2][3] = {{1,0,centerX}, {0,1,centerY}};
  float preM[3][3] = {{1,0,-centerX}, {0,1,-centerY}, {0,0,1}};
  
  float tM[3][3]  = {{1, 0, vparams[0]}, {0, 1, vparams[1]}, {0,0,1}};
  float rM[3][3]  = {{cos(vparams[2]), -sin(vparams[2]), 0}, {sin(vparams[2]), cos(vparams[2]), 0}, {0, 0, 1}};
  float sM[3][3]  = {{exp(vparams[3]), 0, 0}, {0, exp(vparams[3]), 0}, {0, 0, 1}};
  
  CvMat tCVM, rCVM, sCVM, *xform, postCVM, preCVM;
  tCVM  = cvMat(3, 3, CV_32FC1, tM);
  rCVM  = cvMat(3, 3, CV_32FC1, rM);
  sCVM  = cvMat(3, 3, CV_32FC1, sM);

  postCVM = cvMat(2, 3, CV_32FC1, postM);
  preCVM = cvMat(3, 3, CV_32FC1, preM);
  
  xform = cvCreateMat(2, 3, CV_32FC1);
  cvMatMul(&postCVM, &tCVM, xform);
  cvMatMul(xform, &rCVM, xform);
  cvMatMul(xform, &sCVM, xform);
  cvMatMul(xform, &preCVM, xform);
  
  int height = (signed)originalFeats.size();
  int width  = (signed)originalFeats[0].size();
  
  for(int i=0; i<(signed)newFIDs.size(); i++)
    {
      int j = randLocs[i].first, k = randLocs[i].second;
      int nx = (int)(xform->data.fl[0]*k + xform->data.fl[1]*j + xform->data.fl[2] + 0.5f);
      int ny = (int)(xform->data.fl[3]*k + xform->data.fl[4]*j + xform->data.fl[5] + 0.5f);
      if(!(ny >= 0 && ny < height && nx >= 0 && nx < width))
	newFIDs[i] = uniformDist;
      else
	newFIDs[i] = originalFeats[ny][nx];
    }
  
  cvReleaseMat(&xform);
}

float dist(vector<float> &a, vector<float> &b)
{
  float r=0;
  for(int i=0; i<(signed)a.size(); i++)
    r+=(a[i]-b[i])*(a[i]-b[i]);
  return r;
}

void computeGaussian(vector<vector<float> > &Gaussian, int windowSize)
{
  for(int i=0; i<2*windowSize; i++)
    {
      vector<float> grow(2*windowSize);
      for(int j=0; j<2*windowSize; j++)
	{
	  float ii = i-(windowSize-0.5f), jj = j-(windowSize-0.5f);
	  grow[j] = exp(-(ii*ii+jj*jj)/(2*windowSize*windowSize));
	}
      Gaussian.push_back(grow);
    }
}

void getSIFTdescripter(vector<float> &descripter, vector<vector<float> > &m, vector<vector<float> > &theta, int x, int y, int windowSize, int histDim, int bucketsDim, 
		       vector<vector<float> > &Gaussian)
{
  for(int i=0; i<(signed)descripter.size(); i++)
    descripter[i]=0;
  
  int histDimWidth = 2*windowSize/histDim;
  float degPerBin = 360.0f/bucketsDim;
  
  // weight magnitudes by Gaussian with sigma equal to half window
  vector<float> mtimesGRow(2*windowSize);
  vector<vector<float> > mtimesG(2*windowSize, mtimesGRow);
  for(int i=0; i<2*windowSize; i++)
    {
      for(int j=0; j<2*windowSize; j++)
	{
	  int xx = x+i-(windowSize-1), yy = y+j-(windowSize-1);
	  mtimesG[i][j] = m[xx][yy] * Gaussian[i][j];
	}
    }
  
  // calculate descripter
  // using trilinear interpolation
  int histBin[2], histX[2], histY[2];
  float dX[2], dY[2], dBin[2];
  for(int i=0; i<2*windowSize; i++)
    {
      for(int j=0; j<2*windowSize; j++)
	{
	  histX[0] = i/histDim; histX[1] = i/histDim;
	  histY[0] = j/histDim; histY[1] = j/histDim;
	  dX[1] = 0;
	  dY[1] = 0;
	  
	  int iModHD = i % histDim;
	  int jModHD = j % histDim;
	  int histDimD2 = histDim/2;
	  
	  if( iModHD >= histDimD2 && i < 2*windowSize - histDimD2 )
	    {
	      histX[1] = histX[0] + 1;
	      dX[1] = (iModHD + 0.5f - histDimD2) / histDim;
	    }
	  if( iModHD < histDimD2 && i >= histDimD2 )
	    {
	      histX[1] = histX[0] - 1;
	      dX[1] = (histDimD2 + 0.5f - iModHD) / histDim;
	    }
	  if( jModHD >= histDimD2 && j < 2*windowSize - histDimD2 )
	    {
	      histY[1] = histY[0] + 1;
	      dY[1] = (jModHD + 0.5f - histDimD2) / histDim;
	    }
	  if( jModHD < histDimD2 && j >= histDimD2)
	    {
	      histY[1] = histY[0] - 1;
	      dY[1] = (histDimD2 + 0.5f - jModHD) / histDim;
	    }
	  
	  dX[0] = 1.0f - dX[1];
	  dY[0] = 1.0f - dY[1];
	  
	  float histAngle = theta[x+i-(windowSize-1)][y+j-(windowSize-1)];
	  
	  histBin[0] = (int)(histAngle / degPerBin);
	  histBin[1] = (histBin[0]+1) % bucketsDim;
	  dBin[1] = (histAngle - histBin[0]*degPerBin) / degPerBin;
	  dBin[0] = 1.0f-dBin[1];
	  
	  for(int histBinIndex=0; histBinIndex<2; histBinIndex++)
	    {
	      for(int histXIndex=0; histXIndex<2; histXIndex++)
		{
		  for(int histYIndex=0; histYIndex<2; histYIndex++)
		    {
		      int histNum = histX[histXIndex]*histDimWidth + histY[histYIndex];
		      int bin = histBin[histBinIndex];
		      descripter[histNum*bucketsDim + bin] += (mtimesG[i][j] * dX[histXIndex] * dY[histYIndex] * dBin[histBinIndex]);
		    }
		}
	    }
	}
    }
  
  // normalize
  // threshold values at .2, renormalize
  float sum = 0;
  for(int i=0; i<(signed)descripter.size(); i++)
    sum += descripter[i];
  
  if(sum < .0000001f)
    {
      float dn = 1.0f / (signed)descripter.size();
      for(int i=0; i<(signed)descripter.size(); i++)
	descripter[i] = 0;
      return;
    }
  
  for(int i=0; i<(signed)descripter.size(); i++)
    {
      descripter[i] /= sum;
      if(descripter[i] > .2f)
	descripter[i] = .2f;
    }
  sum = 0;
  for(int i=0; i<(signed)descripter.size(); i++)
    sum += descripter[i];
  for(int i=0; i<(signed)descripter.size(); i++)
    descripter[i] /= sum;
}

float findPrincipalAngle(float a1, float v1, float a2, float v2, float a3, float v3)
{
  float A[3][3] = {{a1*a1, a1, 1}, {a2*a2, a2, 1}, {a3*a3, a3, 1}};
  float b[3][1] = {{v1}, {v2}, {v3}};
  
  CvMat Amat = cvMat(3,3,CV_32FC1, A), bmat = cvMat(3,1,CV_32FC1, b), *solmat = cvCreateMat(3, 1, CV_32FC1);
  cvSolve(&Amat, &bmat, solmat);
  
  float r=0;
  if(solmat->data.fl[0] < .0000001f)
    r = a2;
  else
    r = -solmat->data.fl[1] / (2.0f*solmat->data.fl[0]);
  
  cvReleaseMat(&solmat);
  return r;
}

void setRandLocs(vector<pair<int, int> > &randLocs, int h, int w, int paddingH, int paddingW, bool nonRand)
{
  if(!nonRand)
    {
      for(int i=0; i<(signed)randLocs.size(); i++)
	{
	  randLocs[i].first = (rand()%h) + paddingH;
	  randLocs[i].second = (rand()%w) + paddingW;
	}
    }
  else
    {
      for(int i=0; i<h; i++)
	{
	  for(int j=0; j<w; j++)
	    {
	      randLocs[i*w+j].first = i + paddingH;
	      randLocs[i*w+j].second = j + paddingW;
	    }
	}
    }
}

void reorientM(vector<vector<float> > &m, vector<vector<float> > &newM, 
	       vector<vector<float> > &theta, vector<vector<float> > &newTheta,
	       float angle, float cx, float cy, int windowSize)
{
  float Rangle = angle*(pi / 180.0f);
  float rM[3][3]  = {{cos(Rangle), -sin(Rangle), 0}, {sin(Rangle), cos(Rangle), 0}, {0,0,1}};
  float tM[2][3]  = {{1, 0, cy}, {0, 1, cx}};
  
  CvMat tCVM, rCVM, *xform;
  tCVM  = cvMat(2, 3, CV_32FC1, tM);
  rCVM  = cvMat(3, 3, CV_32FC1, rM);
  
  xform = cvCreateMat(2, 3, CV_32FC1);
  cvMatMul(&tCVM, &rCVM, xform);
  
  for(int j=0; j<2*windowSize; j++)
    {
      for(int k=0; k<2*windowSize; k++)
	{
	  float jj = j-windowSize+.5f, kk = k-windowSize+.5f;
	  int nx = (int)(xform->data.fl[0]*kk + xform->data.fl[1]*jj + xform->data.fl[2] + 0.5f);
	  int ny = (int)(xform->data.fl[3]*kk + xform->data.fl[4]*jj + xform->data.fl[5] + 0.5f);
	  	  
	  if(nx < 0) nx = 0;
	  if(nx > (signed)m[0].size()-1) nx = (signed)m[0].size()-1;
	  if(ny < 0) ny = 0;
	  if(ny > (signed)m.size()-1) ny = (signed)m.size()-1;
	  
	  newM[j][k] = m[ny][nx];
	  float tAngle = theta[ny][nx] - angle;
	  if(tAngle < 0)
	    tAngle += 360.0f;
	  newTheta[j][k] = tAngle;
	  
	}
    }
  
  cvReleaseMat(&xform);
}
