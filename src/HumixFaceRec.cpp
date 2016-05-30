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


#include "opencv2/core.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"


#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace cv::face;
using namespace std;


#include <assert.h>
#include <errno.h>

#include "HumixFaceRec.hpp"

#define DEVICE_ID 0


string int2str(int &i) {
	string s;
	stringstream ss(s);
	ss << i;
	return ss.str();
}


//static char* sGetObjectPropertyAsString(
//        v8::Local<v8::Context> ctx,
//        v8::Local<v8::Object> &obj,
//        const char* name,
//        const char* defaultValue) {
//
//    v8::Local<v8::Value> valObj;
//    if ( obj->Get(ctx, Nan::New(name).ToLocalChecked()).ToLocal(&valObj) &&
//            !valObj->IsUndefined() &&
//            !valObj->IsNull()) {
//        v8::String::Utf8Value val(valObj);
//        return strdup(*val);
//    } else {
//        return strdup(defaultValue);
//    }
//}

HumixFaceRec::HumixFaceRec(const v8::FunctionCallbackInfo<v8::Value>& args) 
    : mVideoCap(NULL),
      mCurrentUser(NULL),
      mSaveImgNum(1),
      mTrained(false)
    {
//    v8::Local<v8::Object> config = args[0]->ToObject();
//    v8::Local<v8::Context> ctx = args.GetIsolate()->GetCurrentContext();


    if (!init()) {
        args.GetIsolate()->ThrowException(
                v8::Exception::Error(Nan::New("HumixFaceRec initialization failed").ToLocalChecked()));
        return;
    }

    mState = kStart;
    printf("Humix FaceRec Inited");

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
    mTrainCB.Reset();
    CleanCapturedFaces();
    if (mCurrentUser) {
        free(mCurrentUser);
    }
}

bool HumixFaceRec::init(){
    
    string fn_haar = string("model/lbpcascade_frontalface.xml");
	string fn_eye = string("model/haarcascade_eye.xml");
    
    // Create a FaceRecognizer and train it on the given images:
	//Ptr<FaceRecognizer> model = createFisherFaceRecognizer();
	//Ptr<FaceRecognizer> model = createEigenFaceRecognizer();

    //this model supported train() then update()
    mFacialModel = createLBPHFaceRecognizer();
	
    
	// Create face classifier
	m_haar_cascade.load(fn_haar);
	// Create eye classifier
	m_eye_cascade.load(fn_eye);

    // Holds the current frame from the Video device:
	//Mat frame;
	return true;
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
void HumixFaceRec::sStartCam(const v8::FunctionCallbackInfo<v8::Value>& info) {
    HumixFaceRec* hs = Unwrap<HumixFaceRec>(info.Holder());
    if ( hs == nullptr ) {
        info.GetIsolate()->ThrowException(v8::Exception::ReferenceError(
                Nan::New("Not a HumixFaceRec object").ToLocalChecked()));
        return;
    }

    if ( info.Length() != 1 || !info[0]->IsNumber() ) {
        info.GetIsolate()->ThrowException(v8::Exception::SyntaxError(
                Nan::New("Usage: startCam(<device id>)").ToLocalChecked()));
        return;
    }
    hs->StartCam(info);
}

void
HumixFaceRec::StartCam(const v8::FunctionCallbackInfo<v8::Value>& info) {

    v8::Local<v8::Integer> devInt = info[0].As<v8::Integer>();

    if (mVideoCap != NULL) {
        //restart the capture by delete the old one
        delete mVideoCap;
    }
    // Get a handle to the Video device:
    mVideoCap = new VideoCapture(devInt->Value());
    mVideoCap->set(CV_CAP_PROP_FRAME_WIDTH, 640);
    mVideoCap->set(CV_CAP_PROP_FRAME_HEIGHT, 480);

    if (!mVideoCap->isOpened()) {
        return info.GetReturnValue().Set(false);
    }
    info.GetReturnValue().Set(true);
}

/*static*/
void HumixFaceRec::sStopCam(const v8::FunctionCallbackInfo<v8::Value>& info) {
    HumixFaceRec* hs = Unwrap<HumixFaceRec>(info.Holder());
    if ( hs == nullptr ) {
        info.GetIsolate()->ThrowException(v8::Exception::ReferenceError(
                Nan::New("Not a HumixFaceRec object").ToLocalChecked()));
        return;
    }

    if ( info.Length() != 0 ) {
        info.GetIsolate()->ThrowException(v8::Exception::SyntaxError(
                Nan::New("Usage: stopCam()").ToLocalChecked()));
        return;
    }
    hs->StopCam(info);
}

void
HumixFaceRec::StopCam(const v8::FunctionCallbackInfo<v8::Value>&) {
    if (mVideoCap) {
        delete mVideoCap;
    }
}

/*static*/
void HumixFaceRec::sCaptureFace(const v8::FunctionCallbackInfo<v8::Value>& info) {
    HumixFaceRec* hs = Unwrap<HumixFaceRec>(info.Holder());
    if ( hs == nullptr ) {
        info.GetIsolate()->ThrowException(v8::Exception::ReferenceError(
                Nan::New("Not a HumixFaceRec object").ToLocalChecked()));
        return;
    }

    if ( info.Length() < 1 || info.Length() > 2 || !info[0]->IsString() ) {
        info.GetIsolate()->ThrowException(v8::Exception::SyntaxError(
                Nan::New("Usage: captureFace(string, [string])").ToLocalChecked()));
        return;
    }
    hs->CaptureFace(info);
}

void
HumixFaceRec::CaptureFace(const v8::FunctionCallbackInfo<v8::Value>& info) {

    if (mVideoCap == NULL) {
        info.GetIsolate()->ThrowException(
                v8::Exception::Error(Nan::New("call startCam() first").ToLocalChecked()));
        return;
    }

    Mat frame;
    *mVideoCap >> frame;
    //cap >> frame;
    printf("new frame got \n");
    v8::String::Utf8Value name(info[0]);
    // Convert the current frame to grayscale:
    Mat gray;
    mCurrentSnapshot = frame.clone();

    cvtColor(mCurrentSnapshot, gray, COLOR_BGR2GRAY);
    // Find the faces in the frame:
    vector< Rect_<int> > faces;
    CleanCapturedFaces();
    m_haar_cascade.detectMultiScale(gray, faces);
    for (size_t i = 0; i < faces.size(); i++) {

        printf("new face\n");
        // Process face by face:
        Rect face_i = faces[i];
        // Crop the face from the image. So simple with OpenCV C++:
        Mat face = gray(face_i);

        Mat face_resized;

        // Catch eye
        vector< Rect_<int> > eyes;
        m_eye_cascade.detectMultiScale(face, eyes);
        if (eyes.size() == 2) {
            Mat CutImg = CropFace(mCurrentSnapshot, gray, face_i, eyes);
            cv::resize(CutImg, CutImg, Size(70, 70), 1.0, 1.0, INTER_CUBIC);
            Face* captured = new Face(*name, face_i, CutImg);
            mCapturedFaces.push_back(captured);

            if (info.Length() == 2) {
                v8::String::Utf8Value fileStr(info[1]);
                char num[5];
                sprintf(num, "%lu", i);
                std::string fileName(*fileStr);fileName.append("-").append(num).append(".jpg");
                imwrite(fileName.c_str(), CutImg);
            }
        }
    }

    info.GetReturnValue().Set((unsigned int)mCapturedFaces.size());
}

/*static*/
void HumixFaceRec::sTrainCapturedFace(const v8::FunctionCallbackInfo<v8::Value>& info) {
    HumixFaceRec* hf = Unwrap<HumixFaceRec>(info.Holder());
    if ( hf == nullptr ) {
        info.GetIsolate()->ThrowException(v8::Exception::ReferenceError(
                Nan::New("Not a HumixFaceRec object").ToLocalChecked()));
        return;
    }

    if ( info.Length() != 1 || !info[0]->IsString() ) {
        info.GetIsolate()->ThrowException(v8::Exception::SyntaxError(
                Nan::New("Usage: trainCapturedFace(name)").ToLocalChecked()));
        return;
    }
    hf->TrainCapturedFace(info);
}

void
HumixFaceRec::TrainCapturedFace(const v8::FunctionCallbackInfo<v8::Value>& info) {

    if (mCapturedFaces.size() == 0) {
        info.GetIsolate()->ThrowException(
                v8::Exception::Error(Nan::New("call captureFace() first").ToLocalChecked()));
        return;
    }

    v8::String::Utf8Value name(info[0]);
    std::string nameStr(*name);

    int label = mPersons.size();
    std::map<std::string, int>::iterator it = mPersons.find(nameStr);
    if (it != mPersons.end()) {
        label = it->second;
    }

    for (std::vector<Face*>::iterator it = mCapturedFaces.begin() ; it != mCapturedFaces.end(); ++it) {
        Mat CutImg = (*it)->mMat;
        Rect face = (*it)->mRect;
        // predict person before update
        int prediction = -1;
        double predicted_confidence = 0.0;
        mFacialModel->predict(CutImg, prediction, predicted_confidence);
        rectangle(mCurrentSnapshot, face, Scalar(0, 255, 0), 1);
        stringstream confidence;
        confidence << predicted_confidence;
        std::string predictPerson("");
        for( std::map<std::string,int>::iterator it = mPersons.begin();
                it != mPersons.end(); it++) {
            if (it->second == prediction) {
                predictPerson.append(it->first);
            }
        }
        string box_text = "Id=" + predictPerson + ", conf=" + confidence.str();

        int pos_x = std::max(face.tl().x - 10, 0);
        int pos_y = std::max(face.tl().y - 10, 0);

        // And now put it into the image:
        putText(mCurrentSnapshot, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, Scalar(0, 255, 0), 2);

        // retrain data with CutImg
        //newImages.push_back(CutImg);
        //newLabels.push_back(person.size() - 1);
        mImages.push_back(CutImg);
        vector<Mat> newImages; newImages.push_back(CutImg);
        vector<int> newLabels; newLabels.push_back(label);
        //cout << "newImages.size() = " << newImages.size() << endl;
        if (mTrained) {
            mFacialModel->update(newImages, newLabels);
        } else {
            mFacialModel->train(newImages, newLabels);
            mTrained = true;
        }
        mPersons.insert(std::pair<std::string, int>(nameStr, label));
        cout << "We are push_back label" << label << " , person :" << nameStr << endl;
        // Update
        //model->update(newImages, newLabels);
        // Clear vector
        //newImages.pop_back();
        //newLabels.pop_back();

        // FIXME : add these back
        // Img_Conf << confidence.str() << endl;
        // ID << mPersons.at(prediction) << endl;

        if (predictPerson.compare(nameStr) == 0 && predicted_confidence > 200) {
            cout << "Train success " << endl;
            return info.GetReturnValue().Set(true);
        } else {
            cout << "Id=" << predictPerson << ", conf=" << confidence.str() << endl;
            cout << "Continuing training... " << endl;
        }
    }
    return info.GetReturnValue().Set(false);
}

/*static*/
void HumixFaceRec::sTrain(const v8::FunctionCallbackInfo<v8::Value>& info) {
    HumixFaceRec* hfr = Unwrap<HumixFaceRec>(info.Holder());
    if ( hfr == nullptr ) {
        info.GetIsolate()->ThrowException(v8::Exception::ReferenceError(
                Nan::New("Not a HumixFaceRec object").ToLocalChecked()));
        return;
    }

    if ( info.Length() != 2 || !info[0]->IsString() || !info[1]->IsArray()) {
        info.GetIsolate()->ThrowException(v8::Exception::SyntaxError(
                Nan::New("Usage: train(name, img)").ToLocalChecked()));
        return;
    }

    hfr->Train(info);
}



void
HumixFaceRec::Train(const v8::FunctionCallbackInfo<v8::Value>& info) {

    v8::Isolate* isolate = info.GetIsolate();
    v8::Local<v8::Context> ctx = isolate->GetCurrentContext();

    v8::String::Utf8Value name(info[0]);
    v8::Local<v8::Array> imgArray = info[1].As<v8::Array>();
    std::string nameStr(*name);
    std::map<std::string, int>::iterator it = mPersons.find(nameStr);
    int label = mPersons.size();
    if (it != mPersons.end()) {
        label = it->second;
    }

    vector<Mat> images;
    vector<int> labels;

    uint32_t length = imgArray->Length();
    for (uint32_t index = 0; index < length; index++) {
        v8::String::Utf8Value img(imgArray->Get(ctx, index).ToLocalChecked());
        images.push_back(imread(*img, 0));
        labels.push_back(label);
    }

    if (mTrained) {
        mFacialModel->update(images, labels);
    } else {
        mFacialModel->train(images, labels);
        mTrained = true;
    }
    mPersons.insert(std::pair<std::string, int>(nameStr, label));
}

Mat HumixFaceRec::RotateImage(const Mat source, double angle, int center_x, int center_y, int border)
{
	Mat bordered_source;
	int top, bottom, left, right;
	top = bottom = left = right = border;
	copyMakeBorder(source, bordered_source, top, bottom, left, right, BORDER_CONSTANT, cv::Scalar());
	//Point2f src_center(bordered_source.cols / 2.0F, bordered_source.rows / 2.0F);
	Point2f src_center(center_x, center_y);
	Mat rot_mat = getRotationMatrix2D(src_center, angle, 1.0);
	Mat dst;
	warpAffine(bordered_source, dst, rot_mat, bordered_source.size());
	return dst;
}

Mat HumixFaceRec::CropFace(Mat &orig, Mat &img, Rect &face, vector< Rect_<int> > &eyes) {
	Mat Crop_img;
	int eye_direction[2];
	// Offset percentage
	double offset_pct = 0.2;
	// Deastine size
	int dest_sz = 70;
	// calculate offsets in original image
	double offset_h = floor(float(offset_pct)*dest_sz);
	double offset_v = floor(float(offset_pct)*dest_sz);

    int eye_left[] = {0, 0} ;
    int eye_right[] = {0, 0};
    int temp[] = {0, 0};

	for (size_t j = 0; j < eyes.size(); j++) {
        Rect eyes_j = eyes[j];

        eyes_j.x = eyes_j.x + std::max(face.tl().x, 0) + (eyes_j.width / 2);
        eyes_j.y = eyes_j.y + std::max(face.tl().y, 0) + (eyes_j.height / 2);
        eyes_j.width = 3;
        eyes_j.height = 3;
        rectangle(orig, eyes_j, Scalar(0, 0, 255), 2);

        if (j == 0) {
            temp[0] = eyes_j.x + (eyes_j.width / 2);
            temp[1] = eyes_j.y + (eyes_j.height / 2);
        }
        else {
            //save eye x&y
            if (temp[0] > eyes_j.x + (eyes_j.width / 2)) {
                // first eyes is right eye
                eye_right[0] = temp[0];
                eye_right[1] = temp[1];
                eye_left[0] = eyes_j.x + (eyes_j.width / 2);
                eye_left[1] = eyes_j.y + (eyes_j.height / 2);
            }
            else {
                // second eyes is right eye
                eye_left[0] = temp[0];
                eye_left[1] = temp[1];
                eye_right[0] = eyes_j.x + (eyes_j.width / 2);
                eye_right[1] = eyes_j.y + (eyes_j.height / 2);
            }
        }
    }


	// get the direction
	eye_direction[0] = (eye_right[0] - eye_left[0]);
	eye_direction[1] = (eye_right[1] - eye_left[1]);
	// calc rotation angle in radians
	double rotation;
	if (eye_right[1] - eye_left[1] > 0) {
		rotation = -atan2(float(eye_direction[0]), float(eye_direction[1]));
	}
	else {
		rotation = atan2(float(eye_direction[0]), float(eye_direction[1]));
	}
	// dist between eyes
	double dist = sqrt(eye_direction[1] * eye_direction[1] + eye_direction[0] * eye_direction[0]);
	// calculate the reference eye-width
	double reference = dest_sz - 2.0*offset_h;
	// scale factor
	double scale = float(dist) / float(reference);
	// rotate original around the left eye
	Mat img_rotate = RotateImage(img, rotation, eye_left[0], eye_left[1]);
	//cout << "rotate angle = " << rotation << endl;
	//imwrite(".\\Me\\img_rotate.jpg", img_rotate);
	// Crop the rotate img
	int crop_xy[2];
	crop_xy[0] = eye_left[0] - scale*offset_h + 20;
	crop_xy[1] = eye_left[1] - scale*offset_v + 20;
	Rect cutROI(crop_xy[0], crop_xy[1], (int)(scale*dest_sz), (int)(scale*dest_sz));
	Crop_img = img_rotate(cutROI);
	// Resize Crop_img
	resize(Crop_img, Crop_img, Size(dest_sz, dest_sz), 1.0, 1.0, INTER_CUBIC);
	//imwrite(".\\Me\\Crop_img.jpg", Crop_img);

	return Crop_img;
}


/*static*/
void
HumixFaceRec::sFreeHandle(uv_handle_t* handle) {
    delete handle;
}
/*static*/
v8::Local<v8::FunctionTemplate> HumixFaceRec::sFunctionTemplate(
        v8::Isolate* isolate) {
    v8::EscapableHandleScope scope(isolate);
    v8::Local<v8::FunctionTemplate> tmpl = v8::FunctionTemplate::New(isolate,
            HumixFaceRec::sV8New);
    tmpl->SetClassName(Nan::New("HumixFaceRec").ToLocalChecked());
    tmpl->InstanceTemplate()->SetInternalFieldCount(1);
    NODE_SET_PROTOTYPE_METHOD(tmpl, "startCam", sStartCam);
    NODE_SET_PROTOTYPE_METHOD(tmpl, "stopCam", sStopCam);
    NODE_SET_PROTOTYPE_METHOD(tmpl, "captureFace", sCaptureFace);
    NODE_SET_PROTOTYPE_METHOD(tmpl, "trainCapturedFace", sTrainCapturedFace);
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
