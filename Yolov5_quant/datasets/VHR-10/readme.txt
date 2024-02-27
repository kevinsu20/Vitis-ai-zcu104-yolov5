This very-high-resolution (VHR) remote sensing image dataset was constructed by Dr. Gong Cheng et al. from Northwestern Polytechnical University (NWPU).
This is a 10-class geospatial object detection dataset used for research purposes only.These ten classes of objects are airplane, ship, storage tank, baseball diamond, tennis court, basketball court, ground track field, harbor, bridge, and vehicle.

This dataset contains totally 800 VHR remote sensing images, where the folder "negative image set" includes 150 images that do not contain any targets of the given object classes and the folder "positive image set" includes 650 images with each image containing at least one target to be detected.

These images were cropped from Google Earth and Vaihingen data set and then manually annotated by experts. The Vaihingen data was provided by the German Society for Photogrammetry, Remote Sensing and Geoinformation (DGPF): http://www.ifp.uni-stuttgart.de/dgpf/DKEPAllg.html.

The folder "ground truth" contains 650 separate text files and each one corresponds to an image in "positive image set" folder. Each line of those text files defines a ground truth bounding box in the following format: 
(x1,y1),(x2,y2),a
where (x1,y1) denotes the top-left coordinate of the bounding box, (x2,y2) denotes the right-bottom coordinate of the bounding box, and a is the object class (1-airplane, 2-ship, 3-storage tank, 4-baseball diamond, 5-tennis court, 6-basketball court, 7-ground track field, 8-harbor, 9-bridge, 10-vehicle).

Please cite the following relevant papers when publishing results that use this dataset fully or partly:

Gong Cheng, Junwei Han, Peicheng Zhou, Lei Guo. Multi-class geospatial object detection and geographic image classification based on collection of part detectors. ISPRS Journal of Photogrammetry and Remote Sensing, 98: 119-132, 2014.
Gong Cheng, Junwei Han. A survey on object detection in optical remote sensing images. ISPRS Journal of Photogrammetry and Remote Sensing, 117: 11-28, 2016.
Gong Cheng, Peicheng Zhou, Junwei Han. Learning rotation-invariant convolutional neural networks for object detection in VHR optical remote sensing images. IEEE Transactions on Geoscience and Remote Sensing, 54(12): 7405-7415, 2016.

Contact: gcheng@nwpu.edu.cn;junweihan2010@gmail.com