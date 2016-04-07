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
    static void sV8New(const v8::FunctionCallbackInfo<v8::Value>& info);
    static void sTrain(const v8::FunctionCallbackInfo<v8::Value>& info);
    static void sStart(const v8::FunctionCallbackInfo<v8::Value>& info);
    static void sStop(const v8::FunctionCallbackInfo<v8::Value>& info);

    void Start(const v8::FunctionCallbackInfo<v8::Value>& info);
    void Stop(const v8::FunctionCallbackInfo<v8::Value>& info);
    void Train(const v8::FunctionCallbackInfo<v8::Value>& info);

    static void sReceiveCmd(uv_async_t* handle);
    static void sFreeHandle(uv_handle_t* handle);

    State mState;
    int mArgc;
    char** mArgv;

    v8::Persistent<v8::Function> mCB;

};



#endif /* SRC_HUMIXFACEREC_HPP_ */
