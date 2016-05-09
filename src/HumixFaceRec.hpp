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

class StreamTTS;

class HumixFaceRec : public Nan::ObjectWrap{
public:
    HumixFaceRec(const v8::FunctionCallbackInfo<v8::Value>& args);
    ~HumixFaceRec();

    typedef enum {
        kStart,
        kDetectionMode,
        kTrackingMode,
        kTrainingMode,
        kStop
    } State;

    static v8::Local<v8::FunctionTemplate> sFunctionTemplate(
            v8::Isolate* isolate);
private:
    
    bool init();
    
    static void sV8New(const v8::FunctionCallbackInfo<v8::Value>& info);
    static void sTrain(const v8::FunctionCallbackInfo<v8::Value>& info);
    static void sStart(const v8::FunctionCallbackInfo<v8::Value>& info);
    static void sStop(const v8::FunctionCallbackInfo<v8::Value>& info);
    static void sDetect(const v8::FunctionCallbackInfo<v8::Value>& info);

    void Start(const v8::FunctionCallbackInfo<v8::Value>& info);
    void Stop(const v8::FunctionCallbackInfo<v8::Value>& info);
    void Train(const v8::FunctionCallbackInfo<v8::Value>& info);
    void Detect(const v8::FunctionCallbackInfo<v8::Value>& info);
    
    
    
    static void sTrainLoop(void* arg);
    bool TrainData();
    static void sTrainCompleted(uv_async_t* async);
 
    static void sFreeHandle(uv_handle_t* handle);

    Mat CropFace(Mat img, int *eye_left, int *eye_right);
    Mat RotateImage(const Mat source, double angle, int center_x, int center_y, int border = 20);
    
    State mState;
    int mArgc;
    char** mArgv;

    v8::Persistent<v8::Function> mCB;
    v8::Persistent<v8::Function> mTrainCB;

    uv_thread_t mTrainingThread;

    Ptr<FaceRecognizer> mFacialModel;
    VideoCapture *mVideoCap;

    vector<Mat> mImages;
    vector<int> mLabels;
    // initial training 
    vector<Mat> newImages;
    vector<int> newLabels;
    vector<std::string> mPersons;
    
    char* mCurrentUser;
    int mSaveImgNum;

    CascadeClassifier m_haar_cascade;
	CascadeClassifier m_eye_cascade;
};



#endif /* SRC_HUMIXFACEREC_HPP_ */
