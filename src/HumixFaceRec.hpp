/*******************************************************************************
* Copyright (c) 2015,2016 IBM Corp.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*    http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef SRC_HUMIXFACEREC_HPP_
#define SRC_HUMIXFACEREC_HPP_

#include <string>
#include <queue>
#include <nan.h>
#include <vector>
#include <map>

class StreamTTS;

struct Face {
    Rect mRect;
    Mat mMat;
    char* mName;
    Face(const char* name, Rect rect, Mat mat) {
        mRect = rect;
        mMat = mat;
        mName = strdup(name);
    }
    ~Face() {
        free(mName);
    }
};

class HumixFaceRec : public Nan::ObjectWrap{
public:
    HumixFaceRec(const v8::FunctionCallbackInfo<v8::Value>& args);
    ~HumixFaceRec();

    static v8::Local<v8::FunctionTemplate> sFunctionTemplate(
            v8::Isolate* isolate);
private:
    
    void init();
    
    static void sV8New(const v8::FunctionCallbackInfo<v8::Value>& info);
    static void sStartCam(const v8::FunctionCallbackInfo<v8::Value>& info);
    static void sStopCam(const v8::FunctionCallbackInfo<v8::Value>& info);
    static void sCaptureFace(const v8::FunctionCallbackInfo<v8::Value>& info);
    static void sTrainCapturedFace(const v8::FunctionCallbackInfo<v8::Value>& info);
    static void sDetectCapturedFace(const v8::FunctionCallbackInfo<v8::Value>& info);
    static void sTrain(const v8::FunctionCallbackInfo<v8::Value>& info);

    void StartCam(const v8::FunctionCallbackInfo<v8::Value>& info);
    void StopCam(const v8::FunctionCallbackInfo<v8::Value>& info);
    void CaptureFace(const v8::FunctionCallbackInfo<v8::Value>& info);
    void TrainCapturedFace(const v8::FunctionCallbackInfo<v8::Value>& info);
    void DetectCapturedFace(const v8::FunctionCallbackInfo<v8::Value>& info);
    void Train(const v8::FunctionCallbackInfo<v8::Value>& info);
    
    
//    static void sTrainLoop(void* arg);
//    bool TrainData();
//    static void sTrainCompleted(uv_async_t* async);
 
    static void sFreeHandle(uv_handle_t* handle);

    Mat CropFace(Mat &orig, Mat &gray, Rect &face, vector< Rect_<int> > &eyes);
    Mat RotateImage(const Mat source, double angle, int center_x, int center_y, int border = 20);
    
    void CleanCapturedFaces() {
        while (!mCapturedFaces.empty())
          {
            Face* face = mCapturedFaces.back();
            delete face;
            mCapturedFaces.pop_back();
          }
    }

    void TrainImpl(vector<Mat> &newImages, vector<int> &newLabels) {
        if (mTrained) {
            mFacialModel->update(newImages, newLabels);
        } else {
            mFacialModel->train(newImages, newLabels);
            mTrained = true;
        }
    }

    Ptr<FaceRecognizer> mFacialModel;
    VideoCapture *mVideoCap;

    // initial training 
    std::map<std::string, int> mPersons;

    CascadeClassifier m_haar_cascade;
	CascadeClassifier m_eye_cascade;

	vector<Face*> mCapturedFaces;
	Mat mCurrentSnapshot;
	bool mTrained;
};

#endif /* SRC_HUMIXFACEREC_HPP_ */
