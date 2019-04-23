in progress...


# generate face images
The script keeps generating face images based on DCGAN.  
You can change how long the script produces the images just by changing `for i range():` part.

I trained the Generator using [pytorch tutorial's model][1] with Google Colaboratory.  

run the following code:  
`python generate_face.py`  
then you will see the movie this below.

<video width="320" height="240" controls>
    <source src="movie/generate_face.mov" type="video/mp4">
</video>


# GUI
`show_image.py`

simple GUI that shows an image. The image is loaded as `numpy array`.

## Next
show an image made by Generator

[1]:https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
