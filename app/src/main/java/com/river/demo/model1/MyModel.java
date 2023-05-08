package com.river.demo.model1;
import android.content.Context;
import android.util.Log;
import com.river.demo.OnnxBase;
import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.Collections;
import java.util.Map;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

public class MyModel extends OnnxBase {

    private static final String LOG_TAG = "test";

    private MyModel(){}
    private static class singletonHelper {
        private static final MyModel INSTANCE = new MyModel();
    }
    public static MyModel getInstance() {
        return singletonHelper.INSTANCE;
    }

    public void init(Context context, String modelPath) throws OrtException {
        super.createSession(modelPath);
    }
    private static final int FEATURE_LEN = 22;
    private static final int LABEL_LEN = 400;
    public void runInference(float[] inputData, float[] output1) {
        if (!sessionValid()) {
            Log.d(LOG_TAG, "invalid session!");
            return;
        }

        try {
            Log.d(LOG_TAG, "input: "+ Arrays.toString(inputData));
            long[] inputShape = new long[]{1, FEATURE_LEN};
            OnnxTensor inputTensor = OnnxTensor.createTensor(environment, FloatBuffer.wrap(inputData), inputShape);

            // Run inference
            OrtSession.Result result = session.run(Collections.singletonMap(session.getInputNames().iterator().next(), inputTensor));
            for (Map.Entry<String, OnnxValue> entry : result) {
                Log.d(LOG_TAG, entry.toString());
            }

            float [][] result1 = (float[][])result.get(1).getValue();
            if (result1.length != 1 || result1[0].length != LABEL_LEN) {
                Log.d(LOG_TAG, "invalid result data!");
                return;
            }

            System.arraycopy(result1[0], 0, output1, 0, LABEL_LEN);
            Log.d(LOG_TAG, "output1 is: "+Arrays.toString(output1));

            // Clean up
            inputTensor.close();
            result.close();
        } catch (Exception e) {
            Log.d(LOG_TAG, e.toString());
        }
    }
}


