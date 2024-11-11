package com.example.nativeopencvandroidtemplate

import android.Manifest
import android.app.Activity
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Color
import android.os.Build
import android.os.Bundle
import android.util.Log
import android.view.SurfaceView
import android.view.WindowManager
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import kotlinx.android.synthetic.main.activity_main.*
import org.opencv.android.*
import org.opencv.core.Core
import org.opencv.core.Mat


class MainActivity : Activity(), CameraBridgeViewBase.CvCameraViewListener2 {

    private var mOpenCvCameraView: CameraBridgeViewBase? = null
    private var imageView: ImageView? = null
    private var flagTextView: TextView? = null
    private var startButton: Button? = null
    private var isFindFirstFrame = false
    private var isFindSecondFrame = false
    private var isMatched = false


    private val mLoaderCallback = object : BaseLoaderCallback(this) {
        override fun onManagerConnected(status: Int) {
            when (status) {
                LoaderCallbackInterface.SUCCESS -> {
                    Log.i(TAG, "OpenCV loaded successfully")

                    // Load native library after(!) OpenCV initialization
                    System.loadLibrary("native-lib")

                    mOpenCvCameraView!!.enableView()
                }
                else -> {
                    super.onManagerConnected(status)
                }
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        Log.i(TAG, "called onCreate")
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        // Permissions for Android 6+
        ActivityCompat.requestPermissions(
            this@MainActivity,
            arrayOf(Manifest.permission.CAMERA),
            CAMERA_PERMISSION_REQUEST
        )

        //layout
        setContentView(R.layout.activity_main)
        mOpenCvCameraView = findViewById<CameraBridgeViewBase>(R.id.main_surface)
        mOpenCvCameraView!!.visibility = SurfaceView.VISIBLE
        mOpenCvCameraView!!.setCvCameraViewListener(this)
        imageView = findViewById(R.id.image_view)
        flagTextView = findViewById(R.id.flag_text_view)
        startButton = findViewById(R.id.start_button)


        //listeners
        startButton?.setOnClickListener {
            isFindFirstFrame = !isFindFirstFrame
        }

    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        when (requestCode) {
            CAMERA_PERMISSION_REQUEST -> {
                if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                    mOpenCvCameraView!!.setCameraPermissionGranted()
                } else {
                    val message = "Camera permission was not granted"
                    Log.e(TAG, message)
                    Toast.makeText(this, message, Toast.LENGTH_LONG).show()
                }
            }
            else -> {
                Log.e(TAG, "Unexpected permission request")
            }
        }
    }

    override fun onPause() {
        super.onPause()
        if (mOpenCvCameraView != null)
            mOpenCvCameraView!!.disableView()
    }

    override fun onResume() {
        super.onResume()
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization")
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION, this, mLoaderCallback)
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!")
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        if (mOpenCvCameraView != null)
            mOpenCvCameraView!!.disableView()
    }


    override fun onCameraViewStarted(width: Int, height: Int) {}

    override fun onCameraViewStopped() {}

    override fun onCameraFrame(frame: CameraBridgeViewBase.CvCameraViewFrame): Mat? {
        val imgInput = frame.rgba()
        val img90 = Mat()
        Core.rotate(imgInput, img90, Core.ROTATE_90_CLOCKWISE)
        //native opencv
        var textSend = "Wait to start."
        if (isFindFirstFrame&&!isFindSecondFrame)
            isFindSecondFrame = findBrightImageFromJNI(img90.nativeObjAddr)

        if (isFindSecondFrame)
        {
            isMatched = findMatchesFromJNI(img90.nativeObjAddr)//每次都match
            if(isMatched)
                textSend = "matches found"
            else
                textSend = "matches not found"
        }
        else
            textSend = "Light QR code not found"
        //bitmap show thread
        runOnUiThread {flag_text_view?.text = textSend}
        val bitmap = Bitmap.createBitmap(img90.cols(), img90.rows(), Bitmap.Config.ARGB_8888)
        bitmap.eraseColor(Color.BLACK)
        Utils.matToBitmap(img90, bitmap)
        runOnUiThread { imageView!!.setImageBitmap(bitmap) }
        return null
    }

    private external fun adaptiveThresholdFromJNI(matAddr: Long)
    private external fun findBrightImageFromJNI(matAddr: Long): Boolean
    private external fun findMatchesFromJNI(matAddr: Long): Boolean

    companion object {

        private const val TAG = "MainActivity"
        private const val CAMERA_PERMISSION_REQUEST = 1
    }
}
