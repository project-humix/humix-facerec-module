/*******************************************************************************
 * Copyright (c) 2015 IBM Corp.
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
'use strict';

var console = require('console');
var _ = require('lodash');
var fs = require('fs');
var path = require('path');

var HumixFaceRec = require('./HumixFaceRec').HumixFaceRec;

var imagePath = path.resolve(__dirname, '..', 'images');

module.exports.FaceRec = FaceRec;

function FaceRec(config) {
	this._hfr = new HumixFaceRec(config.options);
	this._hfr.startCam(config.deviceID);
	//use the existing images to train the engine first
	this._existingData = require(config.personImgs);
	this._personImgs = config.personImgs;
	this
	for (var person in this._existingData) {
		console.info('training', person);
		this._hfr.train(person, this._existingData[person]);
	}
}

FaceRec.prototype.train = function(name, images) {
	if (!_.isString(name) || !_.isArray(images)) {
		throw TypeError('train(string, array)');
	}
	this._hfr.train(name, images);
}

FaceRec.prototype.captureAndTrain = function(name, number) {
	if (!_.isString(name) || !_.isNumber(number)) {
		throw new TypeError('captureAndTrain(name, number)');
	}

	console.info('start training mode with user:', name);
	var counter = 0;
	for (var index = 0; index < number; index++) {
		var faces = this._hfr.captureFace(name,
			path.resolve(imagePath, name + counter + '.jpg'));
		if (faces > 0) {
			counter++;
			console.info(faces, 'face(s) captured, trying to associate it with', name);
			this._hfr.trainCapturedFace(name); ///< store original image to ./images/orig/<name>.jpg
			//upload the image below
			fs.renameSync(path.resolve(imagePath, 'orig', name + '.jpg'),
				path.resolve(imagePath, 'orig', 'trained.jpg'));
			console.info('capture again');
			faces = 0;
			for (var dIndex = 0; dIndex < 5 && faces == 0; dIndex++) {
				faces = this._hfr.captureFace(name,
					path.resolve(imagePath, name + counter + '.jpg'));
			}
			if (faces === 0) {
				continue;
			}
			console.info('detect it....');
			var result = this._hfr.detectCapturedFace();
			console.info('result:', result);
			if (result.name === name && result.conf < 70 && result.conf > 0) {
				console.info('trained successfully');
				return true;
			}
		}
	}
	console.info('training failed after', number, 'attempts');
	return false;
}

FaceRec.prototype.detect = function() {
	var faces = 0;
	console.info('detecting face.....');
	for (var dIndex = 0; dIndex < 5 && faces == 0; dIndex++) {
		faces = this._hfr.captureFace('detected',
			path.resolve(imagePath, 'detected.jpg'));
	}
	if (faces === 0) {
		console.info('no face found');
		return;
	}
	console.info('detect it....');
	return this._hfr.detectCapturedFace();

}

FaceRec.prototype.track = function() {

	//console.info('tracking face ....');

	var result = this._hfr.track();
	//console.info('result:' + JSON.stringify(result));

	return result;
}