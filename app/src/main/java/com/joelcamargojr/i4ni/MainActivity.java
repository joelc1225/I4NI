package com.joelcamargojr.i4ni;

import android.Manifest;
import android.annotation.TargetApi;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.databinding.DataBindingUtil;
import android.graphics.Bitmap;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.support.annotation.NonNull;
import android.support.annotation.Nullable;
import android.support.v7.app.AppCompatActivity;
import android.view.View;
import android.widget.Toast;

import com.joelcamargojr.i4ni.databinding.ActivityMainBinding;

public class MainActivity extends AppCompatActivity {

    ActivityMainBinding binding;
    private final static int REQUEST_TAKE_PICTURE = 111;
    private final static int REQUEST_PERMISSIONS = 222;
    boolean arePermissionsGranted;

    @TargetApi(Build.VERSION_CODES.M)
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // Instantiate dataBinding for layout views
        binding = DataBindingUtil.setContentView(this, R.layout.activity_main);

        // Checks if necessary camera permissions are granted before executing action
        if (checkSelfPermission(Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(new String[]{Manifest.permission.CAMERA},
                    REQUEST_PERMISSIONS);
        }

        binding.button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (arePermissionsGranted) {
                    Intent captureImageIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                    startActivityForResult(captureImageIntent, REQUEST_TAKE_PICTURE);
                } else {
                    Toast.makeText(MainActivity.this, "Permission not granted for camera", Toast.LENGTH_LONG).show();

                }
            }
        });
    }

    // Executes once user approves the taken photo
    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        if (requestCode == REQUEST_TAKE_PICTURE) {
            if (resultCode == RESULT_OK) {
                Bitmap bp = (Bitmap) data.getExtras().get("data");
                binding.imageView.setImageBitmap(bp);
            } else if (resultCode == RESULT_CANCELED) {
                Toast.makeText(this, "Cancelled", Toast.LENGTH_LONG).show();
            }
        }
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        if (requestCode == REQUEST_PERMISSIONS) {

            if (grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                arePermissionsGranted = true;
                Toast.makeText(this, "camera permission granted", Toast.LENGTH_LONG).show();

            } else {
                arePermissionsGranted = false;
                Toast.makeText(this, "camera permission denied", Toast.LENGTH_LONG).show();

            }
        }
    }
}
