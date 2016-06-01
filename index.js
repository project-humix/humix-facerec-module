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

var config  = require('./lib/config');
var FaceRec = require('./lib/facerec').FaceRec;

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

var hfr = new FaceRec(config);
hfr.captureAndTrain('yihong', 20);
