# Macros script for inpainting with Firefly in Photoshop automatically
This folder use [Autoit3](https://www.autoitscript.com/site/autoit/downloads/) to edit/run process of inpainting object with Firefly in Photoshop software in Windows machine.

# Install Autoit and requirement libraries
1. Install Autoitv3 for Windows as [link here](https://www.autoitscript.com/site/autoit/downloads/) 
2. Download all libraries [here](https://drive.google.com/drive/folders/1PKF03KAf42Tvk7XgC0BV6nldmkUGvGHy) and paste to folder "**Include**" in Autoit(Path to installed Autoit - Example: C:\Program Files(x86)\AutoIt\Include)
3. Open script "**automation_PTS.au3**" in AutoIt Script and press F5 to Run

# Portable version
If you want to run process immediately, just run "**automation_PTS.exe**"

# Pipeline of inpainting
1. Open image file > Select All > Transform Selection > Fullfill box information > Expand bounding box.
2. Inpaint with 2 methods:
    - Generative Fill:
        - Run generative with empty input prompt will make Firefly remove object instead of fill new content in selected region.
        - Generative fill will return 3 results.
        - GPT Validation(Optional): Save each generation > use ChatGPT to justificate and find the best appropriate gerneration. If this option is disable, the first gerneration will be use for next step.
    - Content-aware fill: Return only 1 result for selected region.
3. Repeat step 2 for each text/object box and save results to output folder.
3. Save final result.

# Preprocess parameters
Before run script, we need define some keyboard shortcuts and parameters below:
## OpenAI API key for ChatGPT Validation
Please paste OpenAI API key into script "**automation_PTS.au3**" here
```bash
Global $apiKey = "" 
```

## Photoshop parameters
1. Keyboard shortbut: Select "**Edit**" > "**Keyboard ShortCut**" (Edit shortcut here)
    - "**Quick save as PNG**": Select "**File**" > save "**Quick save as PNG**" as "**Shift + Ctrl + Alt + Q**".
    - "**Generative fill**":  Select "**Edit**" > save "**Generative fill**" as  "**Shift + Ctrl + Alt + G**"
    - "**Content-aware fill**": Need define some paramter below and save as an action with shortcut "**Shift+F3**":
        - “**Sampling Area Option**”: “Auto”
        - "**Fill Settings**":
            - “**Color Adaption**”: “High”
            - “**Rotation Adaptation**”: “None”
    - "**Expand selection**": Select "**Select**" > save "**Expand**" as  "**Shift + Ctrl + Alt + F1**"
2. Disable some function/window:
    - "Keep ratio between width and height of selection box": Select "**Select**" > Transform selection > Disable button between box "**W**" and box "**H**".
    - Expand "**Generative fill**" validation window enough to show all 3 generations.
3. Open and save 1 image into folder we want to save to make Photoshop know input/output folder.

## Automation script parameters
Because each Window machine will display PTS screen in different resolution, so we need pre-define coordinates for some parameters below before start script:
1. "**X Coordinate:**" : Click "**Find X**" > Open Photoshop > Ctrl+A > Press "**T**" > Click into box "**X**" in the top of PTS options bar.
2. "**Space Coordinate:**": This parameter to make process click into empty space to escape "**Transform selection**"/"**Generative Fill**". Select "**Find Space**" > Click into empty space that do not contain any button/param in PTS screen.
3. "**Folder Images:**": Select "**Browse**" > Choose input folder contain input datasets: image/json file/visualize image.
4. "**Wait Time (ms):**": Define Delay time for waiting "**Gernerative fill**"/"**Content-aware fill**" finish. Time save in milisecond.(Recommend 12000 - 18000 ms)
5. "**Select Fill Method:**": Select method "**Gernerative fill**" or "**Content-aware fill**".
6. "**GPT Validation**": This option available for "**Gernerative fill**" only. Enable this if you want to use ChatGPT detect the best inpaint output.
7. "**Generative Validations Coor:**": Available if "**GPT Validation**" is enable. You need define coordinates of 3  "**Generative fill**" generations. Click button "**First**" and click into position where the first generation is (From left to right). Repeat action for "**Second**" and "**Third**" button.
8. "**Expand pixels:**": Define number of pixels we want to expend selected box.
9. "**Save Intermediate Steps:**": This option will save image for each removed asset. This option available for both inpainting methods if "**GPT Validation**" is disable.
10. "**Start**": Click to start process.