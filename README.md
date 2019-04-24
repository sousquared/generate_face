in progress...


# generate face images
The script keeps generating face images based on DCGAN.  
You can change how long the script produces the images just by changing `for i range():` part.

I trained the Generator using [pytorch tutorial's model][1] with Google Colaboratory.  

run the following code:  
`python generate_face.py`  

# GUI
- `show_image.py`  
simple GUI that just shows an image. The image is loaded as `numpy array`.

- `show_face.py`  
A GUI that generates and shows a face image.

## Next
- update `show_face.py`  
make a button to change the face image.

[1]:https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html