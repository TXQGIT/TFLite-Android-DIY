package com.example.tflite_pure;

import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Matrix;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.EditText;

import java.io.IOException;

/** All the code based on the TFLite demo and the Guide of Android Studio--"Build your first APP"*/

/** Write on 2018/10/20 */

public class MainActivity extends AppCompatActivity {

    public static final String EXTRA_MESSAGE = "com.example.myfirstapp.MESSAGE";

    /** instance of ImageClassifier class*/
    private ImageClassifier classifier;
    private static final String TAG = "TFLitePure";
    public String classifyResult = "Init.";

    /** The root Path of Model, label and image file*/
    private static final String rootPath = "/storage/emulated/0/DCIM/";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        /** Load the model and labels. */
        try {
            // Init the instance classifier
            classifier = new ImageClassifierFloatMobileNet(rootPath);
            classifier.setUseNNAPI(true); /** I do not sure about when to call this method*/
        } catch (IOException e) {
            Log.e(TAG, "Failed to initialize an image classifier.", e);
        }
    }

    /** Is This Right? */
    @Override
    public void onDestroy() {
        classifier.close();
        super.onDestroy();
    }

    /** Called when the user taps the Run button */
    public void sendMessage(View view) {
        Intent intent = new Intent(this, DisplayMessageActivity.class);
        EditText editText = (EditText) findViewById(R.id.editText);
        String message = editText.getText().toString();
        String fileName = rootPath + message;
        classifyImage(fileName);
        intent.putExtra(EXTRA_MESSAGE, classifyResult);
        startActivity(intent);
    }

    /** Resize the size of input image to what the CNN model required. */
    public Bitmap ResizeBitmap(Bitmap bm, int newWidth , int newHeight){
        // 获得图片的宽高.
        int width = bm.getWidth();
        int height = bm.getHeight();
        // 计算缩放比例.
        float scaleWidth = ((float) newWidth) / width;
        float scaleHeight = ((float) newHeight) / height;
        // 取得想要缩放的matrix参数.
        Matrix matrix = new Matrix();
        matrix.postScale(scaleWidth, scaleHeight);
        // 得到新的图片.
        Bitmap newbm = Bitmap.createBitmap(bm, 0, 0, width, height, matrix, true);
        return newbm;
    }

    /** Classifies a image. Return the top3 label and the corresponding prob. */
    private void classifyImage(String fileName) {
        if (classifier == null) {
            classifyResult =  "Uninitialized Classifier or invalid context.";
            return;
        }
        Bitmap bitmap = BitmapFactory.decodeFile(fileName);
        if(bitmap == null){
            classifyResult = "The image is missing!";
            return;
        }
        bitmap = ResizeBitmap(bitmap, classifier.getImageSizeX(), classifier.getImageSizeY());
        classifyResult = classifier.classifyFrame(bitmap);
    }

}
