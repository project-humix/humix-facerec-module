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

var fs = require('fs');
var nats = require('nats').connect();
var HumixSense = require('node-humix-sense');
var path = require('path');
var _ = require('lodash');

var config = require('./lib/config');
var FaceRec = require('./lib/facerec').FaceRec;

var moduleConfig = {
  moduleName:'humix-facerec',
  commands : ['train', 'detect'],
  events : ['detectComplete', 'faceDetected', 'trainComplete'],
  debug: true
}

var humix = new HumixSense(moduleConfig);
var hsm;
var hfr = new FaceRec(config);
var trainedImg = path.resolve(__dirname, 'images', 'orig', 'trained.jpg');
var origImgRoot = path.resolve(__dirname, 'images', 'orig');
var detecting;

var detectedHistory = {};

humix.on('connection', function(humixSensorModule){

  hsm = humixSensorModule;
  console.log('Communication with humix-sense is now ready.');

  hsm.on('train', function(data){
  	data = JSON.parse(data);
  	if (hfr.captureAndTrain(data.name, 20)) {
  	  //read ./images/orig/trained.jpg
  	  var image = fs.readFileSync(trainedImg).toString('base64');
  		hsm.event('trainComplete', 
  		    JSON.stringify({id:data.id, image: image}));
  	} else {
  	  //failed to perform the training
      hsm.event('trainComplete',
          JSON.stringify({id:data.id, msg: 'failed'}));
  	}
  });

  hsm.on('detect', function(data) {
    data = JSON.stringify(data);
    //try detect 3 times
    var result;
    for(var tries = 0; tries < 3; tries++) {
      if (_.isUndefined(result)) continue;
      var img = fs.readFileSync(path.resolve(origImgRoot, result.name + '.jpg')).toString('base64');
      hsm.event('detectComplete', 
          JSON.stringify({id: data.id, name: result.name, conf: result.conf, image: img}));
    }
  })
  
  detecting = setInterval(doDetect, 1000);
});

function doDetect() {
  var result = hfr.detect();
  if (!_.isUndefined(result)) {
    //see if we detect this person within 30 seconds
    try {
      var historyEntry = detectedHistory[result.name];
      if (_.isUndefined(historyEntry) || (Date.now() - historyEntry) > 30000) {
        detectedHistory[result.name] = Date.now();
        var img = fs.readFileSync(path.resolve(origImgRoot, result.name + '.jpg')).toString('base64');
        hsm.event('faceDetected', 
            JSON.stringify({name: result.name, conf: result.conf, image: img}));
      }      
    } catch (e) {
      console.error(e);
    }
  }
}
