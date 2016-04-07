{
  "targets": [
    {
      "target_name": "HumixFaceRec",
      "sources": [
        "./src/HumixFaceRec.cpp"
      ],
      "include_dirs": [ "<!(node -e \"require('nan')\")",
      ],
      "libraries": [ "-Wl",
      ]
    },
    {
      "target_name": "action_after_build",
      "type": "none",
      "dependencies": [ "HumixFaceRec" ],
      "copies": [{
        "destination": "./lib/",
        "files": [
          "<(PRODUCT_DIR)/HumixFaceRec.node"
        ]},
      ]
    }

  ]
}
