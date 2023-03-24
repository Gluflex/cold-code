 using UnityEngine;
 using System.Collections;
 
 public class HiResScreenShots : MonoBehaviour {

    // seting the resolution as same as stream
     public int resWidth = 1920; 
     public int resHeight = 1080;
     public Camera cam;
 
     private bool takeHiResShot = false;

//This script was originally designed to capture a screenshot every second for object detection on an onnx model
//on board the Magic Leap 2.

         void Start()
    {
        //InvokeRepeating("TakeHiResShot", 0.0f, 1.0f);
    }
 
     public void TakeHiResShot() {
         takeHiResShot = true;
     }
 
     void LateUpdate() 
     {
      RenderTexture rt = new RenderTexture(resWidth, resHeight, 24);
      cam.targetTexture = rt;
      Texture2D screenShot = new Texture2D(resWidth, resHeight, TextureFormat.RGB24, false);
      cam.Render();
      RenderTexture.active = rt;
      screenShot.ReadPixels(new Rect(0, 0, resWidth, resHeight), 0, 0);
      cam.targetTexture = null;
      RenderTexture.active = null; // JC: added to avoid errors
      Destroy(rt);
      byte[] bytes = screenShot.EncodeToPNG();
      string filename = "scrnshot.png";
      System.IO.File.WriteAllBytes(filename, bytes);
      Debug.Log(string.Format("Took screenshot to: {0}", filename));
      takeHiResShot = false;
     }
 }