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

var config  = require('./lib/config');
var HumixFaceRec = require('./lib/HumixFaceRec').HumixFaceRec;

/*
var nats    = require('nats').connect();
var HumixSense = require('node-humix-sense');

var moduleConfig = {
    "moduleName":"humix-facerec",
    "commands" : ["start","stop","train"],
    "events" : ["detect"],
    "debug": true
}

var humix = new HumixSense(moduleConfig);
var hsm;

humix.on('connection', function(humixSensorModule){

    hsm = humixSensorModule;

    console.log('Communication with humix-sense is now ready.');

    hsm.on('train', function(data){
        console.log('data:'+data);
        startTraining(data);
    });  // end of say command
});

function startTraining(){

    console.log('start trainning');
}

*/
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
	for(var index = 0; index < number; index++) {
		var faces = this._hfr.captureFace(name, './images/' + name + counter + '.jpg');
		if (faces > 0) {
			counter++;
			console.info('capture', faces, 'face(s), trying to associate it with', name);
			this._hfr.trainCapturedFace(name);
			console.info('capture again');
			faces = 0;
			for (var dIndex = 0; dIndex < 3 && faces == 0; dIndex++) {				
				faces = this._hfr.captureFace(name, './images/' + name + counter + '.jpg');
			}
			console.info('detect it....');
			var result = this._hfr.detectCapturedFace();
			console.info('result:', result);
			if (result.name === name && result.conf < 70 && result.conf > 0) {
				console.info('trained successfully');
				return;
			}
		}
	}
    console.info('training failed after', number, 'attempts');
}

/*
 	console.info('start training mode with user:', name);
	var counter = 0;
	for(var index = 0; index < 10; index++) {
		var faces = this._hfr.captureFace(name, 'images/' + name + counter + '.jpg');
		if (faces > 0) {
			counter++;
			console.info('capture', faces, 'face(s), trying to detect it');
			if (this._hfr.detectFace() === name) {
				console.info('successfully recognize', name);
			}
		}
	}
 */

var hfr = new FaceRec(config);

hfr.captureAndTrain('yihong', 20);
