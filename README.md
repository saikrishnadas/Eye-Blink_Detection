A computer vision application that is capable of detecting and counting blinks in video streams using facial landmarks and OpenCV.
To build our blink detector, we’ll be computing a metric called the eye aspect ratio (EAR).
For more info on this visit: http://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf

EAR = ||p2-p6|| + ||p3-p5|| ÷ 2||p1-p4||

