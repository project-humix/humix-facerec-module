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

//#include "FaceRec.cpp"

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

string int2str(int &i) {
	string s;
	stringstream ss(s);
	ss << i;
	return ss.str();
}


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

HumixFaceRec::HumixFaceRec(const v8::FunctionCallbackInfo<v8::Value>& args) 
    : mTrainingThread(0),
      mCurrentUser(NULL),
      mSaveImgNum(1)
    {
    v8::Local<v8::Object> config = args[0]->ToObject();
    v8::Local<v8::Context> ctx = args.GetIsolate()->GetCurrentContext();


    init();

    mState = kStart;
    printf("Humix FaceRec Inited");
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
    mTrainCB.Reset();
}



static void read_csv(const string& filename, 
                     vector<Mat>& images, 
                     vector<int>& labels, char separator = ';') {
    
    //'
	std::ifstream file(filename.c_str(), ifstream::in);
	if (!file) {
		string error_message = "No valid input file was given, please check the given filename.";
		CV_Error(Error::StsBadArg, error_message);
	}
	string line, path, classlabel;
	while (getline(file, line)) {
		stringstream liness(line);
		getline(liness, path, separator);
		getline(liness, classlabel);
		if (!path.empty() && !classlabel.empty()) {
			images.push_back(imread(path, 0));
			labels.push_back(atoi(classlabel.c_str()));
		}
	}
}

bool HumixFaceRec::init(){
    
      
    // Temporarily hard-code these values
    string fn_csv = string("defaultImg.txt");
    string fn_haar = string("model/lbpcascade_frontalface.xml");
	string fn_eye = string("model/haarcascade_eye.xml");
	
     
    mPersons.push_back("tom");
    mPersons.push_back("croony");
    mPersons.push_back("joey");
    
    // Create a FaceRecognizer and train it on the given images:
	//Ptr<FaceRecognizer> model = createFisherFaceRecognizer();
	//Ptr<FaceRecognizer> model = createEigenFaceRecognizer();
        
    mFacialModel = createLBPHFaceRecognizer();
	
    
	// Create face classifier
	m_haar_cascade.load(fn_haar);
	// Create eye classifier
	m_eye_cascade.load(fn_eye);
    
    
    
    // Read in the data (fails if no valid input filename is given, but you'll get an error message):
	try {
		read_csv(fn_csv, mImages, mLabels);
	}
	// Origional csv have cloony (10images,label:0)   
	catch (cv::Exception& e) {
		cerr << "Error opening file \"" << fn_csv << "\"."<< endl;
		exit(1);
	}
    
	// Train Default Images
	mFacialModel->train(mImages, mLabels);


	// Get a handle to the Video device:
    
    
    mVideoCap = new VideoCapture(DEVICE_ID);
    mVideoCap->set(CV_CAP_PROP_FRAME_WIDTH, 640);
    mVideoCap->set(CV_CAP_PROP_FRAME_HEIGHT, 480);
    
	if (!mVideoCap->isOpened()) {
        
        printf("Capture Device ID: %d cannot be opened. \n", DEVICE_ID);
		return false;
	}
	
    
    
    return true;
    // Holds the current frame from the Video device:
	//Mat frame;

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

    v8::Isolate* isolate = info.GetIsolate();
    v8::Local<v8::Context> ctx = isolate->GetCurrentContext();
 
    v8::Local<v8::Function> cb = info[0].As<v8::Function>();

    mCB.Reset(isolate, cb);
    
   
    v8::Local<v8::Value> argv[] = { Nan::New("Jeffrey").ToLocalChecked() };
    v8::Local<v8::Function> func = v8::Local<v8::Function>::New(isolate, mCB);
    func->CallAsFunction(ctx, ctx->Global(), 1, argv);
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
    mTrainCB.Reset();
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

/*
    if ( info.Length() != 1 || !info[0]->IsString()) {
        info.GetIsolate()->ThrowException(v8::Exception::SyntaxError(
                Nan::New("Usage: train(filename)").ToLocalChecked()));
        return;
    }
    */
    
    hfr->Train(info);
}


void
HumixFaceRec::Train(const v8::FunctionCallbackInfo<v8::Value>& info) {

    v8::Isolate* isolate = info.GetIsolate();
    v8::Local<v8::Context> ctx = isolate->GetCurrentContext();
 
    v8::Local<v8::Function> cb = info[0].As<v8::Function>();

    mTrainCB.Reset(isolate, cb);
    
    
    
    uv_thread_create(&mTrainingThread, sTrainLoop, this);
    /*
    v8::Local<v8::Value> argv[] = { Nan::New("Jeffrey").ToLocalChecked() };
    v8::Local<v8::Function> func = v8::Local<v8::Function>::New(isolate, mCB);
    func->CallAsFunction(ctx, ctx->Global(), 1, argv);
    */
}

void HumixFaceRec::sTrainLoop(void* arg) {
    HumixFaceRec* _this = (HumixFaceRec*)arg;
    
    _this->mState = kTrainingMode;
    
    
    printf(" in sTrainLoop\n");
   
    if(_this->mCurrentUser){
        delete _this->mCurrentUser;
        _this->mCurrentUser = NULL;
    }
    
    
    _this->mCurrentUser = strdup("Jeffrey");
   
  

    if(_this->TrainData()){
        
        // train new face successfully      
        printf("Train Successfully\n");
 
    }else{
        
        // failed to train new face
        printf("Train Failed\n");   
    }
   
    // notify the main thread
    
    uv_async_t* async = new uv_async_t;
    async->data = _this;
    uv_async_init(uv_default_loop(), async, HumixFaceRec::sTrainCompleted);
    uv_async_send(async);
    
    
    _this->mState = kDetectionMode;
    /*
    Mat frame;
	for (;;) {
		_this->mVideoCap >> frame;
		Mat original = frame.clone();
		// Training mode
		if (_this->mState == kTrainingMode) {
			
				if(train_data(original, model, haar_cascade, eye_cascade, person)){
                    
                    //async update
                    
                }

			
		}
    }
    */
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


Mat HumixFaceRec::CropFace(Mat img, int *eye_left, int *eye_right) {
	Mat Crop_img;
	int eye_direction[2];
	// Offset percentage
	double offset_pct = 0.2;
	// Deastine size
	int dest_sz = 70;
	// calculate offsets in original image
	double offset_h = floor(float(offset_pct)*dest_sz);
	double offset_v = floor(float(offset_pct)*dest_sz);
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
	Rect cutROI(int(crop_xy[0]), int(crop_xy[1]), int(scale*dest_sz), int(scale*dest_sz));
	Crop_img = img_rotate(cutROI);
	// Resize Crop_img
	resize(Crop_img, Crop_img, Size(dest_sz, dest_sz), 1.0, 1.0, INTER_CUBIC);
	//imwrite(".\\Me\\Crop_img.jpg", Crop_img);

	return Crop_img;
}



bool HumixFaceRec::TrainData() {

    
  //  vector<string> person;
     
    printf("about to open cam\n");
    
    
	/*
    VideoCapture cap(DEVICE_ID);
	if (!cap.isOpened()) {
		printf( "Capture Device ID %d cannot be opened.", DEVICE_ID);
		return -1;
	}
     
    cap->set(CV_CAP_PROP_FRAME_WIDTH, 640);
    cap->set(CV_CAP_PROP_FRAME_HEIGHT, 480);
  */
    printf("In Train Data\n");
    Mat frame;
	for (;;) {
         
        printf("new frame\n");
		*mVideoCap >> frame;
        //cap >> frame;
        printf("new frame got \n");

        // Convert the current frame to grayscale:
        Mat gray;
        Mat original = frame.clone();
        
        cvtColor(original, gray, COLOR_BGR2GRAY);
        // Find the faces in the frame:
        vector< Rect_<int> > faces;
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
            // Draw eye
            int eye_left[2];
            int eye_right[2];
            int temp[2];


            if (eyes.size() == 2) {
                for (size_t j = 0; j < eyes.size(); j++) {
                    Rect eyes_j = eyes[j];

                    eyes_j.x = eyes_j.x + std::max(face_i.tl().x, 0) + (eyes_j.width / 2);
                    eyes_j.y = eyes_j.y + std::max(face_i.tl().y, 0) + (eyes_j.height / 2);
                    eyes_j.width = 3;
                    eyes_j.height = 3;
                    rectangle(original, eyes_j, Scalar(0, 0, 255), 2);

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
                Mat CutImg = CropFace(gray, eye_left, eye_right);
                cv::resize(CutImg, CutImg, Size(70, 70), 1.0, 1.0, INTER_CUBIC);

                imwrite("./images/me" + int2str(mSaveImgNum) + ".jpg", CutImg);
                mSaveImgNum++;

                // predict person before update
                int prediction = -1;
                double predicted_confidence = 0.0;
                mFacialModel->predict(CutImg, prediction, predicted_confidence);
                rectangle(original, face_i, Scalar(0, 255, 0), 1);
                stringstream confidence;
                confidence << predicted_confidence;
                string box_text = "Id=" + mPersons.at(prediction) + ", conf=" + confidence.str();

                int pos_x = std::max(face_i.tl().x - 10, 0);
                int pos_y = std::max(face_i.tl().y - 10, 0);

                // And now put it into the image:
                putText(original, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, Scalar(0, 255, 0), 2);



                // retrain data with CutImg 
                //newImages.push_back(CutImg);
                //newLabels.push_back(person.size() - 1);
                mImages.push_back(CutImg);
                //cout << "newImages.size() = " << newImages.size() << endl;
                mLabels.push_back(mPersons.size()-1);
                mFacialModel->train(mImages, mLabels);


                cout << "We are push_back label" << mPersons.size() - 1 << " , person :" << mPersons.at(mPersons.size() - 1) << endl;
                // Update
                //model->update(newImages, newLabels);
                // Clear vector
                //newImages.pop_back();
                //newLabels.pop_back();

                
                // FIXME : add these back
                // Img_Conf << confidence.str() << endl;
                // ID << mPersons.at(prediction) << endl;

                if (prediction == mPersons.size() && predicted_confidence > 200) {
                    cout << "Train success " << endl;
                    return true;
                }
                else if (mSaveImgNum == 10) {
                    cout << "Train 10 times,leave " << endl;
                    mSaveImgNum = 0;
                    return true;
                }
                else {
                    cout << "Id=" << mPersons.at(prediction) << ", conf=" << confidence.str() << endl;
                    cout << "Continuing training... " << endl;
                    return false;
                }
            }
        }
	
    }
    return false;
}

void
HumixFaceRec::sTrainCompleted(uv_async_t* async){
    
     
    printf(" in sTrainCompleted\n");
   
    v8::Isolate* isolate = v8::Isolate::GetCurrent();
    v8::HandleScope scope(isolate);
    v8::Local<v8::Context> ctx = isolate->GetCurrentContext();

    HumixFaceRec* _this = (HumixFaceRec*)async->data;
    if ( !_this->mTrainCB.IsEmpty() ) {
        //uv_mutex_lock(&(_this->mCommandMutex));
       
        v8::Local<v8::Value> argv[] = { Nan::New(_this->mCurrentUser).ToLocalChecked() };
        v8::Local<v8::Function> func = v8::Local<v8::Function>::New(isolate, _this->mTrainCB);
        func->CallAsFunction(ctx, ctx->Global(), 1, argv);
    
        //uv_mutex_unlock(&(_this->mCommandMutex));
    }

    uv_close(reinterpret_cast<uv_handle_t*>(async), HumixFaceRec::sFreeHandle);
    
    
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
