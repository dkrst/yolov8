#!/home/dkrst/.venv/yolov8/bin/python3

#
# Generiram i set slika bez dima za validationDetektiram dim na sekvenci
#
#
import cv2
import numpy as np
from glob import glob
import os
import os.path as path
import sys, getopt

sys.path.append('../../../STRIBOR-dev/SEKVENCE')
from sequence import Sequence

sys.path.append('..')
from ultralytics import YOLO

MODEL_PATH = '../LOCAL/MODELS/joinST_5C-Large/best.pt'

# Detekcija na slici 1920x1024 - preskacem 30 linija gore i 26 dole
IMGSZ = (1024, 1920)
Y0 = 30
Y1 = 1054

class SequenceDetect(Sequence):
    def __init__(self, seq_dir, model_path, verbose=False):
        super(SequenceDetect, self).__init__(seq_dir, verbose, zoom_fact)
        self.model = YOLO(model_path)

    # Racuna temporalnu sliku    
    def updateTemporalFrame(self, frindex, first_set, alpha=0.1):
        wframe = self.frame[:,:,0]/255.0 # Only BLUE channel
        if frindex == 0:    # Prvi frame  prolazu
            if first_set:   # Prvi frame u prvom setu, svi na novu sliku
                self.temporal_frame = np.empty(self.frame.shape,
                                               dtype=np.float32)
                for k in range(3):
                    self.temporal_frame[:,:,k] = wframe
                return  # To je to za prvi put
            else:  # Update dugorocne memorije sa prethodnim prolazom
                self.temporal_frame[:,:,0] = self.temporal_frame[:,:,1]
                
        # Updateam sve
        self.temporal_frame[:,:,1] = (1-alpha) * \
                                     self.temporal_frame[:,:,1] + \
                                     alpha*wframe
        self.temporal_frame[:,:,2] = wframe

    def getTemporalFrame(self):
        tframe=np.empty(self.frame.shape, dtype=np.uint8)
        tframe[:,:,0] = self.temporal_frame[:,:,0]*255
        tframe[:,:,1] = self.temporal_frame[:,:,1]*255
        tframe[:,:,2] = self.temporal_frame[:,:,2]*255
        return tframe

    def showTemporalFrame(self):
        cv2.imshow('TEMPORAL', self.getTemporalFrame())

    def yoloDetect(self):
        join_st = np.concatenate((self.frame, self.getTemporalFrame()[:,:,:2]), 2)[Y0:Y1,:]
        if not (hasattr(self, 'wframe')):
            self.wframe = self.frame.copy()
        
        results = self.model.predict([join_st], imgsz=IMGSZ, conf=0.1, max_det=5)
        for r in results:
            boxes = r.boxes.numpy()
            for b in boxes:
                r = b.xyxy[0].astype(int)
                ul = (r[0], r[1]+Y0) # Upper left
                br = (r[2], r[3]+Y0) # Bottom bight
                cv2.rectangle(self.wframe, ul,  br, (0,0,255))

        
    def processSequence(self):
        if not self.checkParms():   # Provjeravamo ispravnost parametara
            return None

        start_index = self.parms['START_INDEX']
        end_index = self.parms['END_INDEX']
        max_index = self.parms['MAX_INDEX']
        cur_index = start_index

        rollup = False
        if start_index > end_index:
            rollup = True

        self.createSequenceWindow()

        seq_running = True
        while seq_running:
            frindex = 0
            while True:
                self.frame = self.readFrame(cur_index, frindex)
                if self.frame is None:
                    break

                self.mask = self.readMask(cur_index, frindex)
                if self.mask is None:
                    seq_running = False
                    break

                self.updateTemporalFrame(frindex, cur_index==start_index)
               
                if self.verbose:
                    print('frame: %5d, %2d' %(cur_index, frindex))

                self.showMaskOnFrame()
                self.yoloDetect()
                self.showFrame()
                #self.showTemporalFrame()
                
                if cv2.waitKey(1) == 27: # sporo je samo po sebi
                    seq_running = False
                    break
                    
                frindex += 1
                del self.frame, self.mask
                    
            if self.verbose:
                print('---------')

            cur_index += 1
            if cur_index > max_index:
                cur_index = 1
                rollup = False
            if rollup==False and cur_index > end_index:
                seq_running = False

        cv2.destroyAllWindows()


# Direktno startanje iz komandne linije 
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Koristenje: %s -i <input_dir> [ -m model -z zoom -h -v ]' %sys.argv[0])
        exit()

    verbose = False
    seq_dir = None
    model_path = MODEL_PATH
    zoom_fact = 1.0
    
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hvi:m:z:",
                                   ["verbose", "input=","model=","zoom="])
    except getopt.GetoptError:
        print('Koristenje: %s -i <input_dir> [ -m model -z zoom -h -v ]' %sys.argv[0])
        sys.exit(2)
        
    for opt, arg in opts:
        if opt == '-h':
            print('Koristenje: %s -i <input_dir> [ -m model -z zoom -h -v ]' %sys.argv[0])
            sys.exit()
        elif opt in ("-i", "--input"):
            seq_dir = arg
        elif opt in ("-m", "--model"):
            model_path = arg
        elif opt in ("-v", "--verbose"):
            verbose = True

    if seq_dir is None:
        print('Koristenje: %s -i <input_dir> [ -m model -z zoom -h -v ]' %sys.argv[0])
        sys.exit()
            
    seq = SequenceDetect(seq_dir, model_path=model_path, verbose=verbose)
    
    seq.processSequence()

