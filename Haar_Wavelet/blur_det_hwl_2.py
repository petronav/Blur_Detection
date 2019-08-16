import pywt, cv2, argparse
import numpy as np
import logging

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to input image.")
ap.add_argument("-t", "--threshold", type=float, default=35, help="Blurry threshold value.")
ap.add_argument("-mz", "--minzero", type=float, default=0.05, help="MinZero decision threshold value.")
args = vars(ap.parse_args())


logger = logging.getLogger('root')
log_file_name = 'blur_hwl.log'
log_format = "%(filename)s: Line - %(lineno)s, Function - %(funcName)20s() | %(message)s"
logging.basicConfig(filename=log_file_name, level=logging.DEBUG, format=log_format)


def blur_detect(img_path, threshold):
    """
    Input : String (path/to/image), Integer (threshold value)
    Output : Tuple of two floating points 
            (Ratio of Dirac-structure and Astep-structure to all the edges [dirac_astep_all_edge_ratio],
            Ratio of how many Roof-Structure and Gstep-Structure edges are blurred [blur_confident_coeff])
    """
    cv_img = cv2.imread(img_path, 0)
    height, width = cv_img.shape[:2]

    logging.debug("Crop input image to certain height and width such that the new shape can be divisible by 2 thrice.")
    cv_img = cv_img[0:int(height/16)*16, 0:int(width/16)*16]
    
    logging.debug("[Step 1] Performing Haar wavelet transform to the image with new shape upto decomposition level = 3")
    LL1, (LH1, HL1, HH1) = pywt.dwt2(cv_img, 'haar')
    LL2, (LH2, HL2, HH2) = pywt.dwt2(LL1, 'haar') 
    LL3, (LH3, HL3, HH3) = pywt.dwt2(LL2, 'haar')

    logging.debug("[Step 2] Construct the edge map in each scale. Emap_i (k,l) = sqrt(LH_i ** 2 + HL_i ** 2 + HH_i ** 2) where i = 1,2,3")
    Emap1 = np.sqrt(np.power(LH1, 2) + np.power(HL1, 2) + np.power(HH1, 2))
    Emap2 = np.sqrt(np.power(LH2, 2) + np.power(HL2, 2) + np.power(HH2, 2))
    Emap3 = np.sqrt(np.power(LH3, 2) + np.power(HL3, 2) + np.power(HH3, 2))
    sub_band1_height, sub_band1_width = Emap1.shape

    logging.debug("[Step 3] Coarsest sliding window size is 8X8.")
    slide_window1_h, slide_window1_w = 8, 8
    logging.debug("Coarser sliding window size is 4X4.")
    slide_window2_h, slide_window2_w = 4, 4
    logging.debug("The sliding window size in the highest scale is 2X2.")
    slide_window3_h, slide_window3_w = 2, 2

    tot_iter = int((sub_band1_height/slide_window1_h)*(sub_band1_width/slide_window1_w))
    logging.debug(f"Number of edge maps, related to sliding windows size : tot_iter = {tot_iter}")
    Emax1, Emax2, Emax3 = np.zeros((tot_iter)), np.zeros((tot_iter)), np.zeros((tot_iter))
    count = 0
    
    x1, y1, x2, y2, x3, y3 = [0]*6
    hori_slide_window_limit = sub_band1_width - slide_window1_w
    logging.debug(f"Sliding windows limit in the horizontal direction, hori_slide_window_limit = {hori_slide_window_limit}")
    
    while count < tot_iter:
        logging.debug("Get the maximum value of slicing windows over edge maps in each level")
        Emax1[count] = np.max(Emap1[x1 : x1+slide_window1_h, y1 : y1+slide_window1_w])
        Emax2[count] = np.max(Emap2[x2 : x2+slide_window2_h, y2 : y2+slide_window2_w])
        Emax3[count] = np.max(Emap3[x3 : x3+slide_window3_h, y3 : y3+slide_window3_w])
        logging.debug("If sliding window ends horizontal direction, move along vertical direction and reset horizontal direction.")
        if y1 == hori_slide_window_limit:
            x1 += slide_window1_h
            x2 += slide_window2_h
            x3 += slide_window3_h
            y1, y2, y3 = 0, 0, 0            
        else:
            y1 += slide_window1_w
            y2 += slide_window2_w
            y3 += slide_window3_w
        count += 1
    logging.debug("[Rule 1] If Emax1(k, l) > threshold or Emax2(k, l) > threshold or Emax3(k, l) > threshold; (k, l) is an edge point.")
    edge_point1 = Emax1 > threshold
    edge_point2 = Emax2 > threshold
    edge_point3 = Emax3 > threshold
    edge_points = edge_point1 + edge_point2 + edge_point3
    n_edges = edge_points.shape[0]
    
    logging.debug("[Rule 2] For an edge point (k, l), if Emax1(k, l) > Emax2(k, l) > Emax3(k, l), (k, l) is Dirac structure or Astep structure.")
    dirac_astep_structure = (Emax1[edge_points] > Emax2[edge_points]) * (Emax2[edge_points] > Emax3[edge_points])

    logging.debug("[Step 4]\n[Rule 3] For any edge point (k, l), if Emax1(k, l) < Emax2(k, l) < Emax3(k, l), (k, l) is Roof-Structure or Gstep-Structure.")
    roof_gstep_structure = (Emax1[edge_points] < Emax2[edge_points]) * (Emax2[edge_points] < Emax3[edge_points])
    logging.debug("[Rule 4] For any edge point (k, l), if Emax2(k, l) > Emax1(k, l) and Emax2(k, l) > Emax3(k, l), (k, l) is Roof-Structure.")
    roof_structure = (Emax2[edge_points] > Emax1[edge_points]) * (Emax2[edge_points] > Emax3[edge_points])
    logging.debug("[Step 5]\n[Rule 5] For any Gstep-Structure or Roof-Structure edge point (k, l), if Emax1(k, l) < threshold, (k, l) is more likely to be in a blurred image.")
    probable_blur = np.zeros(n_edges)

    for i in range(n_edges):
        if roof_gstep_structure[i] == 1 or roof_structure[i] == 1:
            if Emax1[i] < threshold:
                probable_blur[i] = 1                        
        
    logging.debug("[Step 6] Calculate ratio of Dirac-structure and Astep-structure to all the edges, if ratio > minzero then image is unblurred and vice versa.")
    dirac_astep_all_edge_ratio = np.sum(dirac_astep_structure)/np.sum(edge_points)
    
    logging.debug("[Step 7] Calculate how many Roof-Structure and Gstep-Structure edges are blurred.")
    blur_confident_coeff = 100 if np.sum(roof_structure) == 0 else np.sum(probable_blur)/np.sum(roof_structure)
    
    return dirac_astep_all_edge_ratio, blur_confident_coeff


if __name__ == '__main__':
    try:
        per, blur_coeff = blur_detect(args["image"], args["threshold"])
        blur_check = per < args["minzero"]
        print(f"\n\tImage location : {args['image']}\n\tIs blur : {'Yes' if blur_check else 'No'}")
    except Exception as e:
        logging.debug(f"Exception encountered : {e}")