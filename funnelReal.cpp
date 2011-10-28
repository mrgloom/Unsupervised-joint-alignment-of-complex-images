// funnelReal.cpp : funneling for complex, realistic images
//                  using sequence of distribution fields learned from congealReal

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

// usage : funnelReal <list of image filenames> <model file from congealing> <output directory or list of output filenames> [options]
//
//         <list of image filenames> is a list of filenames of the images to process
//         <model file from congealing> is the file containing the sequence of distribution fields from congealing
//         <output directory or list of output filenames> if this is the name of a directory, the aligned images
//            will be written to this directory (making the assumption to the filenames provided in the first
//            argument are relative.  If it is not the name of the directory, then it should be the name of a
//            file containing the filenames to use for the aligned images, in order corresponding to the first argument
//
//         options :
//
//            -o filename
//               output the final parameter values used to generate aligned images
//
//            -outer w h
//               resize images to w by h for funneling computations (default 150x150)
//               this must match the values used in congealing
//
//            -inner w h
//               use an inner window of size w by h, within which to calculate likelihood for congealing
//               (must be smaller than outer dimensions by at least the size of the window for which
//               SIFT descriptor is calculated over) (default 100x100)
//               this must match the values used in congealing

#include "math.h"
#include <cv.h>
#include <cxcore.h>
#include <highgui.h>
#include <assert.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

using namespace std;

void computeGaussian(vector<vector<float> > &Gaussian, int windowSize);
float dist(vector<float> &a, vector<float> &b);
void getSIFTdescripter(vector<float> &descripter, vector<vector<float> > &m, 
		       vector<vector<float> > &theta, int x, int y, int windowSize, int histDim, int bucketsDim, 
		       vector<vector<float> > &Gaussian);
float computeLogLikelihood(vector<vector<float> > &logDistField, vector<vector<float> > &fids, int numFeatureClusters);
void getNewFeatsInvT(vector<vector<float> > &newFIDs, vector<vector<vector<float> > > &originalFeats,
		     vector<float> &vparams, float centerX, float centerY,
		     vector<pair<int, int> > &randPxls);
void showResults(vector<IplImage *> &images, vector<vector<float> > &v, int h, int w,
		 int paddingH, int paddingW, vector<string> &ofns);

const float pi = (float)3.14159265;

int main(int argc, char *argv[])
{
  if(argc < 4)
    {
      cout << "usage: " << argv[0] << " <list of image filenames> <model file from congealing> <output directory or list of output filenames> [options]" << endl;
      return -1;
    }

  const int numParams = 4; // similarity transforms - x translation, y translation, rotation, uniform scaling
  const int windowSize = 4;
  const int maxProcessAtOnce = 600; // set based on memory limitations

  int argcIndex = 4;
  bool outputParams = false;
  ofstream outfile;
  int outerDimW = 150, outerDimH = 150, innerDimW = 100, innerDimH = 100;
  
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
		outfile.open(argv[argcIndex+1]);
		if(!outfile.is_open())
		  {
		    cout << "could not open " << argv[argcIndex+1] << " for writing" << endl;
		    return -1;
		  }
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
	default:
	  cout << "unrecognized option" << endl;
	  return -1;
	}
    }

  ifstream imageList(argv[1]);
  string oDirectory = argv[3];

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

  ifstream trainingInfo(argv[2]);
  int numFeatureClusters, edgeDescDim;
  trainingInfo >> numFeatureClusters >> edgeDescDim;

  vector<float> cRow(edgeDescDim, 0);
  vector<vector<float> > centroids(numFeatureClusters, cRow);
  vector<float> sigmaSq(numFeatureClusters);

  for(int i=0; i<numFeatureClusters; i++)
    {
      for(int j=0; j<edgeDescDim; j++)
	{
	  trainingInfo >> centroids[i][j];
	}
      trainingInfo >> sigmaSq[i];
    }

  int numRandPxls;
  trainingInfo >> numRandPxls;
  vector<pair<int, int> > randPxls(numRandPxls);
  for(int j=0; j<numRandPxls; j++)
    trainingInfo >> randPxls[j].first >> randPxls[j].second;

  vector<float> dfCol(numFeatureClusters, 0);
  vector<vector<float> > logDistField(numRandPxls, dfCol);
  vector<vector<vector<float> > > logDFSeq;

  int iteration;
  while(true)
    {
      trainingInfo >> iteration;
      if(trainingInfo.eof())
	break;
      
      for(int j=0; j<numRandPxls; j++)
	{
	  for(int i=0; i<numFeatureClusters; i++)
	    trainingInfo >> logDistField[j][i];
	}
      logDFSeq.push_back(logDistField);
    }

  vector<string> filenames;
  while(true)
    {
      string t;
      imageList >> t;
      if(imageList.eof())
	break;
      filenames.push_back(t);
    }
  imageList.close();

  string cmd = "test -d " + oDirectory;
  bool isDir = !system(cmd.c_str());
  ifstream ofnsfile;
  if(!isDir)
    {
      ofnsfile.open(oDirectory.c_str());
      if(!ofnsfile.is_open())
	{
	  cout << "could not open " << oDirectory << " for reading" << endl;
	  return -1;
	}
    }

  const int siftHistDim = 4;
  const int siftBucketsDim = 8;
  const int siftDescDim = (4*windowSize*windowSize*siftBucketsDim)/(siftHistDim*siftHistDim);

  vector<vector<float> > Gaussian;
  computeGaussian(Gaussian, windowSize);

  for(int rindex = 0; rindex < filenames.size(); rindex += maxProcessAtOnce)
    {
      vector<IplImage *> originalImages;
      vector<IplImage *> baseImages;
      vector<string> ofns;

      int ulim = min(rindex + maxProcessAtOnce, (int)filenames.size());
      for(int i=rindex; i<ulim; i++)
	{
	  string ofn;
	  IplImage *tmp = cvLoadImage(filenames[i].c_str());
	  if(tmp == 0)
	    {
	      cout << "couldn't find " << filenames[i] << endl;
	      return -1;
	    }
	  if(isDir)
	    ofn = oDirectory + "/" + filenames[i];
	  else
	    ofnsfile >> ofn;
	  ofns.push_back(ofn);
	  
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
	  
	  originalImages.push_back(tmp);
	  baseImages.push_back(tmp3);
	  
	  cvReleaseImage(&tmp2);
	}
      
      int numImages = (signed)baseImages.size();
      vector<float> vrow(numParams);
      vector<vector<float> > v(numImages, vrow);
      
      const int height = baseImages[0]->height-2*windowSize;
      const int width = baseImages[0]->width-2*windowSize;
      const int baseWidthStep = baseImages[0]->widthStep / sizeof(float);
            
      vector<float> ofEntry(edgeDescDim, 0);
      vector<vector<float> > ofCol(width, ofEntry);
      vector<vector<vector<float> > > ofRow(height, ofCol);
      vector<vector<vector<vector<float> > > > originalFeatures(numImages, ofRow);
      vector<float> SiftDesc(edgeDescDim); 
      
      vector<float> mtRow(width+2*windowSize);
      vector<vector<float> > m(height+2*windowSize, mtRow);
      vector<vector<float> > theta(height+2*windowSize, mtRow);
      float dx, dy;
      
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
		  originalFeatures[i][j][k] = SiftDesc;
		}
	    }  
	}
      
      for(int i=0; i<numImages; i++)
	{
	  for(int j=0; j<height; j++)
	    {
	      for(int k=0; k<width; k++)
		{
		  vector<float> distances(numFeatureClusters);
		  float sum = 0;
		  for(int ii=0; ii<numFeatureClusters; ii++)
		    {
		      distances[ii] = exp(-dist(originalFeatures[i][j][k], centroids[ii])/(2*sigmaSq[ii]))/sqrt(sigmaSq[ii]);
		      sum += distances[ii];
		    }
		  for(int ii=0; ii<numFeatureClusters; ii++)
		    distances[ii] /= sum;
		  originalFeatures[i][j][k] = distances;
		}
	    }
	}  
      
      for(int i=0; i<(signed)baseImages.size(); i++)
	cvReleaseImage(&baseImages[i]);
      
      vector<float> fidsEntry(numFeatureClusters, 0);
      vector<vector<float> > fidsRow(numRandPxls, fidsEntry);
      vector<vector<vector<float> > > featureIDs(numImages, fidsRow);
      
      vector<float> nfEntry(numFeatureClusters, 0);
      vector<vector<float> > newFIDs(numRandPxls, nfEntry);
      float centerX = width/2.0f, centerY = height/2.0f;
      
      float d[numParams] = {1.0f, 1.0f, pi/180.0f, 0.02f};
      
      for(int i=0; i<numImages; i++)
	getNewFeatsInvT(featureIDs[i], originalFeatures[i], v[i], centerX, centerY, randPxls);
      
      for(int iter=0; iter<logDFSeq.size(); iter++)
	{
	  for(int j=0; j<numImages; j++)
	    {
	      float oldL = computeLogLikelihood(logDFSeq[iter], featureIDs[j], numFeatureClusters);
	      for(int k=0; k<numParams; k++)
		{
		  float dn = ((rand()%160)-80)/100.0f;
		  if(k>1)
		    dn /= 100.0f;
		  v[j][k] += (d[k] + dn);
		  
		  getNewFeatsInvT(newFIDs, originalFeatures[j], v[j], centerX, centerY, randPxls);
		  float newL = computeLogLikelihood(logDFSeq[iter], newFIDs, numFeatureClusters);
		  
		  if(newL > oldL)
		    {
		      featureIDs[j] = newFIDs;
		      oldL = newL;
		    }
		  else
		    {
		      v[j][k] -= (2*(d[k] + dn));
		      getNewFeatsInvT(newFIDs, originalFeatures[j], v[j], centerX, centerY, randPxls);
		      newL = computeLogLikelihood(logDFSeq[iter], newFIDs, numFeatureClusters);
		      
		      if(newL > oldL)
			{
			  oldL = newL;
			  featureIDs[j] = newFIDs;
			}
		      else
			{
			  v[j][k] += (d[k]+dn);
			}
		    }
		}
	    }
	}
      
      if(outputParams)
	{
	  for(int j=0; j<numImages; j++)
	    {
	      for(int k=0; k<numParams; k++)
		outfile << v[j][k] << " ";
	      outfile << endl;
	    }
	}
      
      showResults(originalImages, v, outerDimH, outerDimW, paddingH, paddingW, ofns);

      for(int i=0; i<numImages; i++)
	cvReleaseImage(&originalImages[i]);
    }
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

float dist(vector<float> &a, vector<float> &b)
{
  float r=0;
  for(int i=0; i<(signed)a.size(); i++)
    r+=(a[i]-b[i])*(a[i]-b[i]);
  return r;
}

void getNewFeatsInvT(vector<vector<float> > &newFIDs, vector<vector<vector<float> > > &originalFeats, 
		     vector<float> &vparams, float centerX, float centerY, vector<pair<int, int> > &randPxls)
{
  int numFeats = newFIDs[0].size();
  vector<float> uniformDist(numFeats, 1.0f/numFeats);
  
  float postM[2][3] = {{1,0,centerX}, {0,1,centerY}};
  float preM[3][3] = {{1,0,-centerX}, {0,1,-centerY}, {0,0,1}};

  float tM[3][3]  = {{1, 0, vparams[0]}, {0, 1, vparams[1]}, {0,0,1}};
  float rM[3][3]  = {{cos(vparams[2]), -sin(vparams[2]), 0}, {sin(vparams[2]), cos(vparams[2]), 0}, {0, 0, 1}};
  float sM[3][3]  = {{exp(vparams[3]), 0, 0}, {0, exp(vparams[3]), 0}, {0, 0, 1}};
  
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

  int height = (signed)originalFeats.size();
  int width  = (signed)originalFeats[0].size();
  
  for(int i=0; i<(signed)newFIDs.size(); i++)
    {
      int j = randPxls[i].first, k = randPxls[i].second;
      int nx = (int)(xform->data.fl[0]*k + xform->data.fl[1]*j + xform->data.fl[2] + 0.5f);
      int ny = (int)(xform->data.fl[3]*k + xform->data.fl[4]*j + xform->data.fl[5] + 0.5f);
      if(!(ny >= 0 && ny < height && nx >= 0 && nx < width))
	newFIDs[i] = uniformDist;
      else
	newFIDs[i] = originalFeats[ny][nx];
    }
  
  cvReleaseMat(&xform);
}

float computeLogLikelihood(vector<vector<float> > &logDistField, vector<vector<float> > &fids, int numFeatureClusters)
{
  float l = 0;
  for(int j=0; j<(signed)fids.size(); j++)
    {
      for(int i=0; i<numFeatureClusters; i++)
	l += fids[j][i] * logDistField[j][i];
    }
  return l;
}

void showResults(vector<IplImage *> &images, vector<vector<float> > &v, int h, int w,		 
		 int paddingH, int paddingW, vector<string> &ofns)
{
  for(int i=0; i<images.size(); i++)
    {
      float cropT1inv[2][3] = {{1,0,images[i]->width/2.0f}, {0,1,images[i]->height/2.0f}};
      float cropS1inv[3][3] = {{images[i]->width/(float)w,0,0}, {0,images[i]->height/(float)h,0}, {0,0,1}};		    
      float cropS2inv[3][3] = {{w/(float)images[i]->width,0,0}, {0,h/(float)images[i]->height,0}, {0,0,1}};		    
      float cropT2inv[3][3] = {{1,0,-images[i]->width/2.0f}, {0,1,-images[i]->height/2.0f}, {0,0,1}};
   
      float postM[3][3] = {{1,0,w/2.0f}, {0,1,h/2.0f}, {0,0,1}};
      float preM[3][3] = {{1,0,-w/2.0f}, {0,1,-h/2.0f}, {0,0,1}};
      
      float tM[3][3]  = {{1, 0, v[i][0]}, {0, 1, v[i][1]}, {0,0,1}};
      float rM[3][3]  = {{cos(v[i][2]), -sin(v[i][2]), 0}, {sin(v[i][2]), cos(v[i][2]), 0}, {0, 0, 1}};
      float sM[3][3]  = {{exp(v[i][3]), 0, 0}, {0, exp(v[i][3]), 0}, {0, 0, 1}};
      
      CvMat tCVM, rCVM, sCVM, hxCVM, hyCVM, *xform, postCVM, preCVM, cropT1invCVM, cropS1invCVM, cropS2invCVM, cropT2invCVM;
      
      tCVM  = cvMat(3, 3, CV_32FC1, tM);
      rCVM  = cvMat(3, 3, CV_32FC1, rM);
      sCVM  = cvMat(3, 3, CV_32FC1, sM);
      
      postCVM = cvMat(3, 3, CV_32FC1, postM);
      preCVM = cvMat(3, 3, CV_32FC1, preM);

      cropT1invCVM = cvMat(2,3,CV_32FC1, cropT1inv);
      cropS1invCVM = cvMat(3,3,CV_32FC1, cropS1inv);
      cropS2invCVM = cvMat(3,3,CV_32FC1, cropS2inv);
      cropT2invCVM = cvMat(3,3,CV_32FC1, cropT2inv);
      
      xform = cvCreateMat(2, 3, CV_32FC1);
      cvMatMul(&cropT1invCVM, &cropS1invCVM, xform);
      cvMatMul(xform, &tCVM, xform);
      cvMatMul(xform, &rCVM, xform);
      cvMatMul(xform, &sCVM, xform);
      cvMatMul(xform, &cropS2invCVM, xform);
      cvMatMul(xform, &cropT2invCVM, xform);     
      
      IplImage* dst =  cvCreateImage(cvSize(images[i]->width, images[i]->height), images[i]->depth, 3);
      cvWarpAffine(images[i], dst, xform, CV_WARP_INVERSE_MAP + CV_WARP_FILL_OUTLIERS + CV_INTER_LINEAR);
      
      cvSaveImage(ofns[i].c_str(), dst);

      cvReleaseMat(&xform);      
      cvReleaseImage(&dst);
    }      
}
