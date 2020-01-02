# EE838A_HW4
Homework 4, Advanced Image Restoration and Quality Enhancement, EE, KAIST, Fall 2018

1. Explanation
	- All video .mp4 files are stored in the folder './data/60fps'
	- All training frames are saved in the folder './data/train'
	- All validation frames are stored in the folder './data/valid'
	- The model trained with only L1 loss is saved in the folder './report/model_L1_loss/model'
	- The model trained with combined loss is saved in the folder './report/model_L1_perceptual_loss/model'
	- Read './report/HW4_20184187_DinhVu_Report.pdf' for more explanation

2. Create the training frames
	- Copy 5 video files .mp4 to the folder './data/60fps'
	- Open Matlab
	- Browse the working space in Matlab to the folder './source'

	2.1. Extract Frames
	
		- In Command Window of Matlab, type: ExtractFrame
		- Wait the process finished, it may take several hours
		- When the process finished, you can see the extracted frame in the folders './data/60fps/videoN' where N = 1,2,3,4,5

	2.2. Scene Separation
	
		- In Command Window of Matlab, type: SceneSeparation
		- Wait the process finished, it may take several hours
		- When the process finished, you can see the frames is distributed into difference scene in the folder './data/train/videoN' where N = 1,2,3,4,5

3. Training
	- Open Command Promt (Windows) or Terminal (Linux) in the folder './source'
	- Type: python main.py --mode=train
	- Wait the training process finished, it may take 1 or 2 days
	- The model is saved in the folder './model'
	- All logs are written in the file './logs/logs_train.txt'

4. Validation
	- If you want to run L1 loss model, copy all files in the folder './report/model_L1_loss/model' to './model'
	- If you want to run combined loss model, copy all file in the folder './report/model_L1_perceptual_loss/model' to './model'
	- Open Command Promt (Windows) or Terminal (Linux) in the folder './source'
	- Type: python main.py --mode=valid
	- Wait the validation process finished, it may take few minutes
	- The interpolation frames are stored in the folder './report/valid_intp'
	- All logs are written in the file './logs/logs_test.txt'

	- If you just want to see the interplolation frames without doing the above steps:
		- For L1 loss model:
			+ All interplolation frames are saved in the folder './report/model_L1_loss/valid_intp'
			+ All qualified criteria are written in the file './report/model_L1_loss/logs/logs_valid.txt'

		- For combined loss model:
			+ All interplolation frames are saved in the folder './report/model_L1_perceptual_loss/valid_intp'
			+ All qualified criteria are written in the file './report/model_L1_perceptual_loss/logs/logs_valid.txt'

5. Testing
	- Make the structure of your testing data same as validation data
	- If you want to run L1 loss model, copy all files in the folder './report/model_L1_loss/model' to './model'
	- If you want to run combined loss model, copy all file in the folder './report/model_L1_perceptual_loss/model' to './model'
	- Open Command Promt (Windows) or Terminal (Linux) in the folder './source'
	- Type: python main.py --mode=test
	- Wait the testing process finished, it may take time depend on how large your test data
	- All logs are written in the file './logs/logs_test.txt'



