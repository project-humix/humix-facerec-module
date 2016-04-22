/*
* Copyright (c) 2011. Philipp Wagner <bytefish[at]gmx[dot]de>.
* Released to public domain under terms of the BSD Simplified license.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above copyright
*     notice, this list of conditions and the following disclaimer in the
*     documentation and/or other materials provided with the distribution.
*   * Neither the name of the organization nor the names of its contributors
*     may be used to endorse or promote products derived from this software
*     without specific prior written permission.
*
*   See <http://www.opensource.org/licenses/bsd-license>
*/

#include "opencv2/core.hpp"
#include "opencv2/face.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <time.h>
#include <math.h>

using namespace cv;
using namespace cv::face;
using namespace std;

string int2str(int &i);


// Open .txt file for save new_pic's name & Conf. & id
ofstream Img_Conf;
ofstream ID;
// Initial save Img_frame number & time
int Save_Img_num = 1;
int Count = 0;
int temp_prediction;
int Count_same_face = 0;
bool succcess = false;

// Mode 0: detect ; Mode 1: training mode
int mode = 0;

// These vectors hold the images and corresponding labels:
vector<Mat> images;
vector<int> labels;
// initial training 
vector<Mat> newImages;
vector<int> newLabels;


Mat rotateImage(const Mat source, double angle, int center_x, int center_y, int border = 20)
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


Mat Crop_face(Mat img, int *eye_left, int *eye_right) {
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
	Mat img_rotate = rotateImage(img, rotation, eye_left[0], eye_left[1]);
	//cout << "rotate angle = " << rotation << endl;
	imwrite(".\\Me\\img_rotate.jpg", img_rotate);
	// Crop the rotate img
	int crop_xy[2];
	crop_xy[0] = eye_left[0] - scale*offset_h + 20;
	crop_xy[1] = eye_left[1] - scale*offset_v + 20;
	Rect cutROI(int(crop_xy[0]), int(crop_xy[1]), int(scale*dest_sz), int(scale*dest_sz));
	Crop_img = img_rotate(cutROI);
	// Resize Crop_img
	resize(Crop_img, Crop_img, Size(dest_sz, dest_sz), 1.0, 1.0, INTER_CUBIC);
	imwrite(".\\Me\\Crop_img.jpg", Crop_img);

	return Crop_img;
}




static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
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

int detect(Mat &original, Ptr<FaceRecognizer> &model, CascadeClassifier haar_cascade, CascadeClassifier eye_cascade, vector<string> person) {
	// Check for valid command line arguments, print usage
	// if no arguments were given.


	// Convert the current frame to grayscale:
	Mat gray;
	cvtColor(original, gray, COLOR_BGR2GRAY);
	// Find the faces in the frame:
	vector< Rect_<int> > faces;
	haar_cascade.detectMultiScale(gray, faces);
	// At this point you have the position of the faces in
	// faces. Now we'll get the faces, make a prediction and
	// annotate it in the video. Cool or what?

	for (size_t i = 0; i < faces.size(); i++) {
		// Process face by face:
		Rect face_i = faces[i];
		// Crop the face from the image. So simple with OpenCV C++:
		Mat face = gray(face_i);
		// Resizing the face is necessary for Eigenfaces and Fisherfaces. You can easily
		// verify this, by reading through the face recognition tutorial coming with OpenCV.
		// Resizing IS NOT NEEDED for Local Binary Patterns Histograms, so preparing the
		// input data really depends on the algorithm used.
		//
		// I strongly encourage you to play around with the algorithms. See which work best
		// in your scenario, LBPH should always be a contender for robust face recognition.
		//
		// Since I am showing the Fisherfaces algorithm here, I also show how to resize the
		// face you have just found:
		Mat face_resized;
		int prediction = -1;
		double predicted_confidence = 0.0;

		// Catch eye
		vector< Rect_<int> > eyes;
		eye_cascade.detectMultiScale(face, eyes);
		// Draw eye
		int eye_left[2];
		int eye_right[2];
		int temp[2];



		if (eyes.size() == 2) {
			for (size_t j = 0; j < eyes.size(); j++) {
				Rect eyes_j = eyes[j];

				//eyes_j.x = eyes_j.x + std::max(face_i.tl().x, 0);
				//eyes_j.y = eyes_j.y + std::max(face_i.tl().y, 0);
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
			Mat CutImg = Crop_face(gray, eye_left, eye_right);

			//cv::resize(CutImg, CutImg, Size(70, 70), 1.0, 1.0, INTER_CUBIC);
			//cv::resize(face, face_resized, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);
			// Now perform the prediction, see how easy that is:
			model->predict(CutImg, prediction, predicted_confidence);
			//model->predict(face_resized, prediction, predicted_confidence);
			// And finally write all we've found out to the original image!
			rectangle(original, face_i, Scalar(0, 255, 0), 1);
			// First of all draw a green rectangle around the detected face:
			//rectangle(original, face_i, Scalar(0, 255, 0), 1);
			// Create the text we will annotate the box with:
			stringstream confidence;
			confidence << predicted_confidence;
			string box_text = "Id=" + person.at(prediction) + ", conf=" + confidence.str();
			// Calculate the position for annotated text (make sure we don't
			// put illegal values in there):
			int pos_x = std::max(face_i.tl().x - 10, 0);
			int pos_y = std::max(face_i.tl().y - 10, 0);

			//Img_Conf << confidence.str() << endl;
			//ID << person.at(prediction) << endl;

			// And now put it into the image:
			putText(original, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, Scalar(0, 255, 0), 2);

			cout << "Id=" << person.at(prediction) << ", conf=" << confidence.str() << endl;

			if (temp_prediction == prediction) {
				if (Count_same_face >= 5) {
					Count_same_face = 0;
					cout << "{'name':" << person.at(prediction) << ",'confidence':" << confidence.str() << "}" << endl;
				}
				else {
					if (Count_same_face >= 2 && predicted_confidence < 200) {
						return 1;
					}
					Count_same_face++;
				}
			}
			else {
				temp_prediction = prediction;
				Count_same_face = 0;
			}
			/*
			if (predicted_confidence < 200) {
			return 1;
			}
			*/

			/*
			if (Count == 5) {
			Count = 0;
			cout << "Add new images & update" << endl;
			//
			// Cut face & save img
			//imwrite(".\\Me\\" + int2str(Save_Img_num) + ".jpg", CutImg);
			//
			//
			// Save origional face
			//resize(face, face, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);
			//imwrite(".\\Me\\" + int2str(Save_Img_num) + ".jpg", face);

			}
			Count++;
			*/


		}
	}

	return 0;

}
bool train_data(Mat &original, Ptr<FaceRecognizer> &model, CascadeClassifier haar_cascade, CascadeClassifier eye_cascade, vector<string> person) {

	// Convert the current frame to grayscale:
	Mat gray;
	cvtColor(original, gray, COLOR_BGR2GRAY);
	// Find the faces in the frame:
	vector< Rect_<int> > faces;
	haar_cascade.detectMultiScale(gray, faces);

	for (size_t i = 0; i < faces.size(); i++) {
		// Process face by face:
		Rect face_i = faces[i];
		// Crop the face from the image. So simple with OpenCV C++:
		Mat face = gray(face_i);

		Mat face_resized;

		// Catch eye
		vector< Rect_<int> > eyes;
		eye_cascade.detectMultiScale(face, eyes);
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
			Mat CutImg = Crop_face(gray, eye_left, eye_right);
			cv::resize(CutImg, CutImg, Size(70, 70), 1.0, 1.0, INTER_CUBIC);

			imwrite(".\\Me\\" + int2str(Save_Img_num) + ".jpg", CutImg);
			Save_Img_num++;

			// predict person before update
			int prediction = -1;
			double predicted_confidence = 0.0;
			model->predict(CutImg, prediction, predicted_confidence);
			rectangle(original, face_i, Scalar(0, 255, 0), 1);
			stringstream confidence;
			confidence << predicted_confidence;
			string box_text = "Id=" + person.at(prediction) + ", conf=" + confidence.str();

			int pos_x = std::max(face_i.tl().x - 10, 0);
			int pos_y = std::max(face_i.tl().y - 10, 0);

			// And now put it into the image:
			putText(original, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, Scalar(0, 255, 0), 2);



			// retrain data with CutImg 
			//newImages.push_back(CutImg);
			//newLabels.push_back(person.size() - 1);
			images.push_back(CutImg);
			//cout << "newImages.size() = " << newImages.size() << endl;
			labels.push_back(person.size()-1);
			model->train(images, labels);


			cout << "We are push_back label" << person.size() - 1 << " , person :" << person.at(person.size() - 1) << endl;
			// Update
			//model->update(newImages, newLabels);
			// Clear vector
			//newImages.pop_back();
			//newLabels.pop_back();

			
			Img_Conf << confidence.str() << endl;
			ID << person.at(prediction) << endl;

			if (prediction == person.size() && predicted_confidence > 200) {
				cout << "Train success " << endl;
				return true;
			}
			else if (Save_Img_num == 10) {
				cout << "Train 10 times,leave " << endl;
				Save_Img_num = 0;
				return true;
			}
			else {
				cout << "Id=" << person.at(prediction) << ", conf=" << confidence.str() << endl;
				cout << "Continuing training... " << endl;
				return false;
			}
		}
	}
	return false;
}

int main(int argc, const char *argv[]) {

	// Create .txt file
	Img_Conf.open("Img_Conf.txt");
	ID.open("ID.txt");

	vector<string> person = { "clooney","Emma" };

	//string fn_haar = string("haarcascade_frontalface_alt_tree.xml");	
	string fn_haar = string("lbpcascade_frontalface.xml");
	string fn_eye = string("haarcascade_eye.xml");
	string fn_csv = string("dataset_ori.txt");

	// Create face classifier
	CascadeClassifier haar_cascade;
	haar_cascade.load(fn_haar);
	// Create eye classifier
	CascadeClassifier eye_cascade;
	eye_cascade.load(fn_eye);
	// Read in the data (fails if no valid input filename is given, but you'll get an error message):
	try {
		read_csv(fn_csv, images, labels);
	}
	// Origional csv have cloony (10images,label:0)   
	catch (cv::Exception& e) {
		cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
		// nothing more we can do
		exit(1);
	}
	// Get the height from the first image. We'll need this
	// later in code to reshape the images to their original
	// size AND we need to reshape incoming faces to this size:
	int im_width = images[0].cols;
	int im_height = images[0].rows;
	// Create a FaceRecognizer and train it on the given images:
	//Ptr<FaceRecognizer> model = createFisherFaceRecognizer();
	Ptr<FaceRecognizer> model = createLBPHFaceRecognizer();
	//Ptr<FaceRecognizer> model = createEigenFaceRecognizer();

	// Origional Train
	model->train(images, labels);

	// Set deviced ID
	int deviceId = 0;
	// Get a handle to the Video device:
	VideoCapture cap(deviceId);
	if (!cap.isOpened()) {
		cerr << "Capture Device ID " << deviceId << "cannot be opened." << endl;
		return -1;
	}
	// Holds the current frame from the Video device:
	Mat frame;
	for (;;) {
		cap >> frame;
		Mat original = frame.clone();
		// Training mode
		if (mode == 1) {
			if (succcess == false) {
				cout << "In training mode " << endl;
				succcess = train_data(original, model, haar_cascade, eye_cascade, person);
				if (succcess == true) {
					mode = 0;
					succcess = false;
				}

			}
		}
		// Detect mode
		if (mode == 0) {
			cout << "In detect mode " << endl;
			mode = detect(original, model, haar_cascade, eye_cascade, person);
			//mode = 0;
			if (mode == 1) {
				cout << "Ready to Train new people " << endl;
				string name;
				cout << "What your name?" << endl;
				cin >> name;
				person.push_back(name);
			}
		}


		// Show the result:
		imshow("face_recognizer", original);
		// And display it:
		char key = (char)waitKey(20);
		// Exit this loop on escape:
		if (key == 27)
			break;

	}
	return 0;

	Img_Conf.close();
	ID.close();
}
string int2str(int &i) {
	string s;
	stringstream ss(s);
	ss << i;
	return ss.str();
}
