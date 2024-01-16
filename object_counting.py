
"""

Input: image from camera, darknet image object, loaded network (darknet object), class_name 
       (darknet object), trck_dict(dictionary with number of cylinders and their respective centroid 
       coordinates),st_dict(number of cylinders in the first frame), count (number of frames),
       cyl(cylinder number), moving(if cylinder is moving then True else False) 

Output: trck_dict(dictionary with number of cylinders and their respective centroid coordinates),
        st_dict(number of cylinders in the first frame), count (number of frames),
        cyl(cylinder number), moving(if cylinder is moving then True else False) 

User Requirement:
1) Detecting if the cylinder are moving or not


Requirements:
1) This function takes the darknet image object, loaded network(darknet object), class name(darknet onject),
   and image from the camera  which is first cropped in Region of inetrest(ROI) and then it is converted to 
   the darknet image object which is passed to the loaded model with class names. The result is the detection 
   of cylinder in each ROI, which basically provides the central coordinates of the bounding box detection of 
   the respective object.
2) Then we need to check if the cylinders are moving or not by the monement of coordinates of the detection over some frames.
3) If there is no movement for certain frames then we know the cylinfers are not moving.
"""


import darknet
import cv2
import error
from datetime import datetime, timedelta
import traceback
import numpy as np
class ObjetCount():
	font = cv2.FONT_HERSHEY_SIMPLEX
	def __init__(self):
		self.is_first =1
		self.total_count =0
		self.count_frame=0
		self.num = 1
		self.current_dict ={}
		self.current_len =0
		self.line1 = 0
		self.line2 = 0
		self.line3 = 0
		self.line3_buffer = 10
		self.trigger = "Bottle"
		self.trigger_confidence = 70
		self.pr_cord = 0
		self.orientation = "horizontal"
	def get_input_data(self,result,img,x_res,y_res):
		#
		if result != []:
			for i,j in enumerate(result):
				cord= j[2]
				conf = j[1]
				cl_name = j[0]
				if cl_name == self.trigger and float(conf) >= self.trigger_confidence:
					cord=j[2]
					xm=int((cord[0]) * float(x_res/self.model_width)) # cent coordinates
					ym=int((cord[1]) * float(y_res/self.model_height))
					#xco=int(float(cord[0]-cord[2]/2) * float(x_res/416)) # bounding box coordinates
					#yco=int(float(cord[1]-cord[3]/2) * float(y_res/416))
					#xExt=int(float(cord[2]) * float(x_res/416))
					#yExt=int(float(cord[3]) * float(y_res/416))
					if self.pr_cord == 1:
						#img=cv2.rectangle(img,(xco,yco),(xco+xExt,yco+yExt),(0,0,255),2)
						img=cv2.rectangle(img,(xm-2,ym-2),(xm+2,ym+2),(0,255,0),2)
					if self.orientation == "horizontal":
						ch = xm
					elif self.orientation == "vertical":
						ch = ym
					#movement is assumed to be left to right in orizontal position
					if self.is_first == 1:
						if (ch >= self.line1) and (ch <=(self.line3 + self.line3_buffer)):
							self.current_dict[self.num]={'xco':xm,'yco':ym,'state':"Initialized"}
							self.is_first =0
					elif self.is_first == 0 and ch >= self.line1 and (ch <=(self.line3 + self.line3_buffer)):

						for key in self.current_dict:
							if self.orientation == "horizontal":
								old_ch = int(self.current_dict[key]['xco'])
							else:
								old_ch = int(self.current_dict[key]['yco'])
							dist = abs(ch - old_ch)
							print ("Distance from previous frame: "+str(dist))
							if dist < self.diff_pixel:
								if ch >= self.line2 and ch <= self.line3 and self.current_dict[key]["state"] == "Initialized":
									self.current_dict[key]["state"] = "counted"
									self.total_count = self.total_count + 1
								if ch >= self.line1 and ch <= self.line2 and self.current_dict[key]["state"] != "counted":
									self.current_dict[key]["state"] = "Initialized"
								self.current_dict[key]['xco'] = xm
								self.current_dict[key]['yco'] = ym
								self.checked = 1

						if self.checked == 0 and ch >=self.line1 and ch <= self.line2:
							self.num = self.num +1
							self.current_dict[self.num]={'xco':xm,'yco':ym,'state':"Initialized"}
		return self.total_count



def track(img,darknet_image,network,class_names,moving):
	global total_count,count_frame,current_dict,prev_dict,current_len,prev_len,is_first
	try:
		print (total_count)
		x_res=int(img.shape[1])
		y_res=int(img.shape[0])
		pts = np.array([[120,450],[120,650],[450,650],[450,450]])
		#pts = np.array([[600,480],[600,690],[950,690],[950,480]])
		mask = np.zeros(img.shape[:2], np.uint8)
		cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
		dst = cv2.bitwise_and(img, img, mask=mask)
		frame_rgb = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
		frame_resized = cv2.resize(frame_rgb,(darknet.network_width(network),darknet.network_height(network)),interpolation=cv2.INTER_LINEAR)
		darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())
		result=darknet.detect_image(network,class_names,darknet_image, thresh=0.25)
		#print(result)
		
		img,count= get_input_data(result,img,x_res,y_res)

		cv2.putText(img, "Count : "+str(count), (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
		#print("moving - > "+str(moving))
		return(img)
	except Exception as e:
		print(str(e))
		traceback.print_exc()