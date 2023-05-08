package com.river.demo;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

public class OnnxBase {
    public OrtEnvironment environment = OrtEnvironment.getEnvironment();
    public OrtSession session;

    public void createSession(String modelPath) throws OrtException {
        if (session == null) {
            OrtSession.SessionOptions options = new OrtSession.SessionOptions();
            options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT);
            session =  environment.createSession(modelPath, options);
        }
    }

    public boolean sessionValid() {
        return session != null;
    }
}
