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

#include <assert.h>
#include <errno.h>

#include "HumixFaceRec.hpp"

/*
static const arg_t cont_args_def[] = {
    POCKETSPHINX_OPTIONS,
    // Argument file. 
    {"-argfile",
     ARG_STRING,
     NULL,
     "Argument file giving extra arguments."},
    {"-adcdev",
     ARG_STRING,
     NULL,
     "Name of audio device to use for input."},
    {"-time",
     ARG_BOOLEAN,
     "no",
     "Print word times in file transcription."},
    {"-cmdproc",
     ARG_STRING,
     "./processcmd.sh",
     "command processor."},
    {"-wav-say",
     ARG_STRING,
     "./voice/interlude/pleasesay1.wav",
     "the wave file of saying."},
    {"-wav-proc",
     ARG_STRING,
     "./voice/interlude/process1.wav",
     "the wave file of processing."},
    {"-wav-bye",
     ARG_STRING,
     "./voice/interlude/bye.wav",
     "the wave file of goodbye."},
    {"-keyword-name",
      ARG_STRING,
      "HUMIX",
      "keyword of the name."},
    {"-lang",
     ARG_STRING,
     "zh-tw",
     "language locale."},

    CMDLN_EMPTY_OPTION
};
*/

static char* sGetObjectPropertyAsString(
        v8::Local<v8::Context> ctx,
        v8::Local<v8::Object> &obj,
        const char* name,
        const char* defaultValue) {

    v8::Local<v8::Value> valObj;
    if ( obj->Get(ctx, Nan::New(name).ToLocalChecked()).ToLocal(&valObj) &&
            !valObj->IsUndefined() &&
            !valObj->IsNull()) {
        v8::String::Utf8Value val(valObj);
        return strdup(*val);
    } else {
        return strdup(defaultValue);
    }
}

HumixFaceRec::HumixFaceRec(const v8::FunctionCallbackInfo<v8::Value>& args){
    v8::Local<v8::Object> config = args[0]->ToObject();
    v8::Local<v8::Context> ctx = args.GetIsolate()->GetCurrentContext();

    mState = kStart;

    //mCMDProc = sGetObjectPropertyAsString(ctx, config, "cmdproc", "./util/processcmd.sh");
    //mWavSay =  sGetObjectPropertyAsString(ctx, config, "wav-say", "./voice/interlude/pleasesay1.wav");

    /*
    char const *cfg;
    v8::Local<v8::Array> props = config->GetPropertyNames();
    int propsNum = props->Length();
    mArgc = propsNum * 2;
    mArgv = (char**)calloc(mArgc, sizeof(char**));
    int counter = 0;
    for ( int i = 0; i < propsNum; i++ ) {
        v8::Local<v8::Value> valObj;
        if ( props->Get(ctx, i).ToLocal(&valObj) ) {
            //option: need to add '-' prefix as an option
            v8::String::Utf8Value name(valObj);
            char** p = mArgv + counter++;
            *p = (char*)malloc(name.length() + 2);
            sprintf(*p, "-%s", *name);
            if ( config->Get(ctx, valObj).ToLocal(&valObj) &&
                    !valObj->IsNull() &&
                    !valObj->IsUndefined()) {
                //option value
                v8::String::Utf8Value val(valObj);
                p = mArgv + counter++;
                *p = strdup(*val);
            }
        }
    }
    mConfig = cmd_ln_parse_r(NULL, cont_args_def, mArgc, mArgv, TRUE);
    */

    Wrap(args.This());
}

HumixFaceRec::~HumixFaceRec() {

    mCB.Reset();
}

/*static*/
void HumixFaceRec::sV8New(const v8::FunctionCallbackInfo<v8::Value>& info) {
    v8::Isolate* isolate = info.GetIsolate();
    if ( info.Length() != 1 ) {
        info.GetIsolate()->ThrowException(
                v8::Exception::SyntaxError(Nan::New("one argument").ToLocalChecked()));
        return info.GetReturnValue().Set(v8::Undefined(isolate));
    }

    v8::Local<v8::Object> configObj = info[0]->ToObject();

    if ( configObj.IsEmpty() ) {
        info.GetIsolate()->ThrowException(
                v8::Exception::SyntaxError(Nan::New("The first argument shall be an object").ToLocalChecked()));
        return info.GetReturnValue().Set(v8::Undefined(isolate));
    }

    new HumixFaceRec(info);
    return info.GetReturnValue().Set(info.This());
}

/*static*/
void HumixFaceRec::sStart(const v8::FunctionCallbackInfo<v8::Value>& info) {
    HumixFaceRec* hs = Unwrap<HumixFaceRec>(info.Holder());
    if ( hs == nullptr ) {
        info.GetIsolate()->ThrowException(v8::Exception::ReferenceError(
                Nan::New("Not a HumixFaceRec object").ToLocalChecked()));
        return;
    }

    if ( info.Length() < 1 || !info[0]->IsFunction() ) {
        info.GetIsolate()->ThrowException(v8::Exception::SyntaxError(
                Nan::New("Usage: start(callback)").ToLocalChecked()));
        return;
    }
    hs->Start(info);
}

void
HumixFaceRec::Start(const v8::FunctionCallbackInfo<v8::Value>& info) {
    v8::Local<v8::Function> cb = info[0].As<v8::Function>();

    mCB.Reset(info.GetIsolate(), cb);
}

/*static*/
void HumixFaceRec::sStop(const v8::FunctionCallbackInfo<v8::Value>& info) {
    HumixFaceRec* hs = Unwrap<HumixFaceRec>(info.Holder());
    if ( hs == nullptr ) {
        info.GetIsolate()->ThrowException(v8::Exception::ReferenceError(
                Nan::New("Not a HumixFaceRec object").ToLocalChecked()));
        return;
    }

    if ( info.Length() != 0 ) {
        info.GetIsolate()->ThrowException(v8::Exception::SyntaxError(
                Nan::New("Usage: stop()").ToLocalChecked()));
        return;
    }
    hs->Stop(info);
}

void
HumixFaceRec::Stop(const v8::FunctionCallbackInfo<v8::Value>& info) {
    mCB.Reset();
    mState = kStop;
}

/*static*/
void HumixFaceRec::sTrain(const v8::FunctionCallbackInfo<v8::Value>& info) {
    HumixFaceRec* hfr = Unwrap<HumixFaceRec>(info.Holder());
    if ( hfr == nullptr ) {
        info.GetIsolate()->ThrowException(v8::Exception::ReferenceError(
                Nan::New("Not a HumixFaceRec object").ToLocalChecked()));
        return;
    }

    if ( info.Length() != 1 || !info[0]->IsString()) {
        info.GetIsolate()->ThrowException(v8::Exception::SyntaxError(
                Nan::New("Usage: train(filename)").ToLocalChecked()));
        return;
    }
    hfr->Train(info);
}


/*static*/
v8::Local<v8::FunctionTemplate> HumixFaceRec::sFunctionTemplate(
        v8::Isolate* isolate) {
    v8::EscapableHandleScope scope(isolate);
    v8::Local<v8::FunctionTemplate> tmpl = v8::FunctionTemplate::New(isolate,
            HumixFaceRec::sV8New);
    tmpl->SetClassName(Nan::New("HumixFaceRec").ToLocalChecked());
    tmpl->InstanceTemplate()->SetInternalFieldCount(1);
    NODE_SET_PROTOTYPE_METHOD(tmpl, "start", sStart);
    NODE_SET_PROTOTYPE_METHOD(tmpl, "stop", sStop);
    NODE_SET_PROTOTYPE_METHOD(tmpl, "train", sTrain);

    return scope.Escape(tmpl);
}


void InitModule(v8::Local<v8::Object> target) {
    v8::Isolate* isolate = v8::Isolate::GetCurrent();
    Nan::HandleScope scope;
    v8::Local<v8::Context> ctx = isolate->GetCurrentContext();

    v8::Local<v8::FunctionTemplate> ft = HumixFaceRec::sFunctionTemplate(isolate);

    target->Set(ctx, Nan::New("HumixFaceRec").ToLocalChecked(),
            ft->GetFunction(ctx).ToLocalChecked());
}

NODE_MODULE(HumixFaceRec, InitModule);
