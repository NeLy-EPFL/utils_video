import cv2 as cv
import sys
import argparse
import os
import shutil


def main():
    # Define the program description
    text = 'You can compress videos through ffmpeg using this script.'
    
    # Initiate the parser with a description
    parser = argparse.ArgumentParser(description=text, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-e", "--encoder", help="Define the encoder used by ffmpeg.", default="libx264")
    parser.add_argument("-c", "--crf", help="Define the crf used by ffmpeg.", default="28")
    parser.add_argument("-f", "--fps", help="Define the fps used by ffmpeg (Only when folder with images is provided with flag -i).")
    parser.add_argument("-p", "--preset", help="Define the preset used by ffmpeg.", default="veryslow")
    parser.add_argument('-k',"--keep", action='store_true', help="If present, the images will be kept.")
    parser.add_argument('-i',"--imgName", help="Images name format. Example: img_%%04d.jpg")
    
    parser.add_argument('video_name')
    
    args = parser.parse_args()
    
    path_split = args.video_name.split('/')
    
    if len(path_split)>1:
        parent_folder = os.getcwd()+'/'+args.video_name[0:args.video_name.find(path_split[-1])]
        vid_name = path_split[-1]
    else:
        parent_folder = os.getcwd()+'/'
        vid_name = args.video_name
    
    
    print("Processing: "+ vid_name)
    
    if not args.imgName:
        in_path = parent_folder+vid_name
        out_path = in_path.replace('.mp4', '_compressed.mp4')
    
        # Open video
        cap = cv.VideoCapture(in_path)
        if cap.isOpened() == False:
            sys.exit('Video file cannot be read! Please check the path to the video or add flag -i if you want to use a sequence of images')
    
        fps = int(cap.get(5))
        imgs = []
    
        print("Reading video...")
        while(True):
            # Capture frame-by-frame
            ret, frame = cap.read()
    
            if ret == True:
                imgs.append(frame)
            else:
                break
    
        tmp_folder = parent_folder + '/imgs_' + vid_name.replace('.mp4','')
    
        if not os.path.exists(tmp_folder):
            os.makedirs(tmp_folder)        
    
        print("Obtaining images...")
        for num, img in enumerate(imgs):
            img_name = tmp_folder + '/img_' + str(num) + '.jpg'
            cv.imwrite(img_name, img)
        imgName = "img_%d.jpg"
    else:
        tmp_folder = parent_folder+vid_name
        out_path = parent_folder+vid_name+".mp4"
        imgName = args.imgName
        if not args.fps:
            sys.exit('Framerate not specified! Use flag -f to define it.')
        else:
            fps = args.fps
        
        
    print("Writing video...")
    terminal_call = "ffmpeg -loglevel error -nostats -r "+ str(fps) + " -i " + tmp_folder + "/" + imgName+ " -c:v " + args.encoder + " -crf " + args.crf + " -preset " + args.preset + " -pix_fmt yuv420p " + out_path    
    os.system(terminal_call)
    
    if not args.keep and not args.imgName:
        shutil.rmtree(tmp_folder)
        
    
    print("Compression done!")

if __name__ == "__main__":
    main()
