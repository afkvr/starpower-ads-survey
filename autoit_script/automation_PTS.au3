#include <JSON.au3>
#include <Misc.au3>
#include <GUIConstantsEx.au3>
#include <WindowsConstants.au3>
#include <Array.au3>
#include <Date.au3>
#include <WindowsConstants.au3>
#include <WinHttp.au3>
#include <EditConstants.au3>
#include "CvtAny2Base64.au3"
#include <AutoItConstants.au3>


Func NormalizePath($folderPath)
    If StringRight($folderPath, 1) = "\" Then
        $folderPath = StringTrimRight($folderPath, 1)
    EndIf
    Return $folderPath
EndFunc

; Global Variables for images list
Global $imageFiles = []


Global $apiKey = ""     ; Paste OpenAI key here

; Variables for mouse coordinates
Global $coordX1 = 190 , $coordY1 = 40
Global $coordX2 = 130 , $coordY2 = 210
Global $genFillCoordinates[3][2] ;Set coordinates for 3 generations of Generative Fill
$genFillCoordinates[0][0] = 1450
$genFillCoordinates[0][1] = 358
$genFillCoordinates[1][0] = 1528
$genFillCoordinates[1][1] = 371
$genFillCoordinates[2][0] = 1616
$genFillCoordinates[2][1] = 367
; Create GUI Window
GUICreate("Automation", 400, 500)

; Labels and Input fields for coordinates
GUICtrlCreateLabel("X Coordinate:", 10, 10)
$coordX1_Input = GUICtrlCreateInput("", 170, 10, 50, 20)
$coordY1_Input = GUICtrlCreateInput("", 230, 10, 50, 20)
$moveButton1 = GUICtrlCreateButton("Find X", 300, 10, 80, 20)

GUICtrlCreateLabel("Space Coordinate:", 10, 50)
$coordX2_Input = GUICtrlCreateInput("", 170, 50, 50, 20)
$coordY2_Input = GUICtrlCreateInput("", 230, 50, 50, 20)
$moveButton2 = GUICtrlCreateButton("Find Space", 300, 50, 80, 20)

; Input for base folder path
GUICtrlCreateLabel("Folder Images:", 10, 100)
$pathInput = GUICtrlCreateInput("", 10, 130, 270, 20)
$browseButton = GUICtrlCreateButton("Browse", 290, 130, 80, 20)

; Input for sleep time
GUICtrlCreateLabel("Wait Time (ms):", 10, 170)
$sleepTimeInput = GUICtrlCreateInput("10000", 170, 170, 100, 20)

; Radio Buttons for Fill Method Selection
GUICtrlCreateLabel("Select Fill Method:", 10, 210)
$generativeFill_Radio = GUICtrlCreateRadio("Generative Fill", 170, 210, 150, 20)
$contentAwareFill_Radio = GUICtrlCreateRadio("Content-Aware Fill", 170, 270, 150, 20)
GUICtrlSetState($generativeFill_Radio, $GUI_CHECKED) ; Default to Generative Fill

;Add GPT Validation
$gptValCheckbox = GUICtrlCreateCheckbox("GPT Validation", 170, 240, 150, 20)

; Labels and Input fields for coordinates
GUICtrlCreateLabel("Generative Validations Coor:", 10, 310)
$coordX1_genFill = GUICtrlCreateInput("", 170, 310, 50, 20)
$coordY1_genFill = GUICtrlCreateInput("", 230, 310, 50, 20)
$moveButton_genFill_1 = GUICtrlCreateButton("First", 300, 310, 80, 20)

$coordX2_genFill = GUICtrlCreateInput("", 170, 335, 50, 20)
$coordY2_genFill = GUICtrlCreateInput("", 230, 335, 50, 20)
$moveButton_genFill_2 = GUICtrlCreateButton("Second", 300, 335, 80, 20)

$coordX3_genFill = GUICtrlCreateInput("", 170, 360, 50, 20)
$coordY3_genFill = GUICtrlCreateInput("", 230, 360, 50, 20)
$moveButton_genFill_3 = GUICtrlCreateButton("Third", 300, 360, 80, 20)

; Define number pixel expanding
GUICtrlCreateLabel("Expand pixels:", 10, 400)
$numberPixelExpand = GUICtrlCreateInput("0", 170, 400, 50, 20)

; Checkbox for saving intermediate steps
GUICtrlCreateLabel("Save Intermediate Steps:", 10, 440)
$saveStepsCheckbox = GUICtrlCreateCheckbox("", 170, 440, 20, 20)

; Start button
$startButton = GUICtrlCreateButton("Start", 150, 480, 120, 30)

GUISetState(@SW_SHOW)

; Function to read a PNG file and convert to Base64
Func ConvertImageToBase64($sFilePath)
    ; Check if the file exists
    If Not FileExists($sFilePath) Then
        MsgBox(0, "Error", "File not found: " & $sFilePath)
        Return
    EndIf
    
    ; Read the binary data from the PNG file
    Local $hFile = FileOpen($sFilePath, 16) ; Open as binary
    Local $sBinaryData = FileRead($hFile)
    FileClose($hFile)
    
    ; Encode the binary data to Base64
    Local $sBase64 = _Base64Encode($sBinaryData, 0) ; 'True' to get it as a string
    
    ; Return the Base64 string
    Return $sBase64
EndFunc

Func ChatGPT_Request($message)
    Local $url = "https://api.openai.com/v1/chat/completions"
    Local $headers = "Content-Type: application/json" & @CRLF & "Authorization: Bearer " & $apiKey
    Local $data = '{"model": "gpt-4o", "messages": [{"role": "user", "content": ' & $message & '}]}'

    Local $oHTTP = ObjCreate("WinHttp.WinHttpRequest.5.1")
    $oHTTP.Open("POST", $url, False)
    $oHTTP.SetRequestHeader("Content-Type", "application/json")
    $oHTTP.SetRequestHeader("Authorization", "Bearer " & $apiKey)
    $oHTTP.Send($data)

    Local $response = $oHTTP.ResponseText
    ;~ SendAndLog("GPT response: " & $response)
    ; Parse the JSON response
    Local $json = _JSON_Parse($response)
    If @error Then
        Return "Error: Failed to parse JSON response"
    EndIf

    ; Extract the message content
    Local $choices = _JSON_Get($json, "choices")
    If IsArray($choices) And UBound($choices) > 0 Then
        Local $messageObj = _JSON_Get($choices[0], "message")
        If IsMap($messageObj) Then
            Local $content = _JSON_Get($messageObj, "content")
            SendAndLog("GPT response: " & $content)
            Local $bestImage = getIndexGPTResponse($content)
            Return $bestImage
        Else
            Return "Error: Message object not found"
        EndIf
    Else
        Return "Error: Choices array not found or empty"
    EndIf
EndFunc

Func getIndexGPTResponse($message)
    ; Find the last number in the string using a regular expression
    Local $aMatches = StringRegExp($message, "\d+", 3)
    ; Check if any numbers were found
    If @error Then
        Return "none"
    Else
        ; Get the last number
        Local $lastNumber = $aMatches[UBound($aMatches) - 1]
        Return $lastNumber
    EndIf
EndFunc

Func SendAndLog($Data, $FileName = -1, $TimeStamp = False)
    If $FileName == -1 Then $FileName = @ScriptDir & '\Log.txt'
    ;~ Send($Data)
    $hFile = FileOpen($FileName, 1)
    If $hFile <> -1 Then
        If $TimeStamp = True Then $Data = _Now() & ' - ' & $Data
        FileWriteLine($hFile, $Data)
        FileClose($hFile)
    EndIf
EndFunc

; Function to detect left mouse click and save coordinates
Func WaitForClick(ByRef $coordX, ByRef $coordY)
    While 1
        If _IsPressed("01") Then
            $mousePos = MouseGetPos()
            $coordX = $mousePos[0]
            $coordY = $mousePos[1]
            ExitLoop
        EndIf
        Sleep(100)
    WEnd
EndFunc

While 1
    $msg = GUIGetMsg()

    Select
        Case $msg = $GUI_EVENT_CLOSE
            Exit

        Case $msg = $moveButton1
            WaitForClick($coordX1, $coordY1)
            GUICtrlSetData($coordX1_Input, $coordX1)
            GUICtrlSetData($coordY1_Input, $coordY1)

        Case $msg = $moveButton2
            WaitForClick($coordX2, $coordY2)
            GUICtrlSetData($coordX2_Input, $coordX2)
            GUICtrlSetData($coordY2_Input, $coordY2)
        
        Case $msg = $moveButton_genFill_1
            WaitForClick($genFillCoordinates[0][0], $genFillCoordinates[0][1])
            GUICtrlSetData($coordX1_genFill, $genFillCoordinates[0][0])
            GUICtrlSetData($coordY1_genFill, $genFillCoordinates[0][1])
            SendAndLog("Coor-1: " & $genFillCoordinates[0][0] & " - " & $genFillCoordinates[0][1])
        Case $msg = $moveButton_genFill_2
            WaitForClick($genFillCoordinates[1][0], $genFillCoordinates[1][1])
            GUICtrlSetData($coordX2_genFill, $genFillCoordinates[1][0])
            GUICtrlSetData($coordY2_genFill, $genFillCoordinates[1][1])
            SendAndLog("Coor-2: " & $genFillCoordinates[1][0] & " - " &  $genFillCoordinates[1][1])
        Case $msg = $moveButton_genFill_3
            WaitForClick($genFillCoordinates[2][0], $genFillCoordinates[2][1])
            GUICtrlSetData($coordX3_genFill, $genFillCoordinates[2][0])
            GUICtrlSetData($coordY3_genFill, $genFillCoordinates[2][1])
            SendAndLog("Coor-3: " & $genFillCoordinates[2][0] & " - " &  $genFillCoordinates[2][1])

        Case $msg = $browseButton
            Local $selectedFolder = FileSelectFolder("Select Folder", "")
            If @error = 0 Then
                GUICtrlSetData($pathInput, $selectedFolder)
            EndIf

        Case $msg = $startButton
            Local $baseFolderPath = GUICtrlRead($pathInput)
            $baseFolderPath = NormalizePath($baseFolderPath)

            Local $sleepTime = GUICtrlRead($sleepTimeInput)

            Local $expandPixels = GUICtrlRead($numberPixelExpand)

            $imageFiles = GetImageFiles($baseFolderPath)
			;~ _ArrayDisplay($imageFiles, "Danh sách ảnh cuối cùng")

            Local $fillMethod = "Generative Fill"
            If GUICtrlRead($contentAwareFill_Radio) = $GUI_CHECKED Then
                $fillMethod = "Content-Aware Fill"
            EndIf

            If UBound($imageFiles) > 0 Then
                ProcessImages($baseFolderPath, $sleepTime, $fillMethod, $gptValCheckbox, $expandPixels)
            Else
                MsgBox(16, "Error", "No valid image files found in the directory. [2]")
            EndIf
    EndSelect

    ; Exit the program with ESC key if needed
    If _IsPressed("1B") Then ; ESC key
        MsgBox(0, "Exit", "Program will exit due to ESC key pressed.")
        Exit
    EndIf
    
    Sleep(100)
WEnd

Func GetImageFiles($folderPath)
    Local $fileList[1]
    $fileList[0] = ""

    ; Check if the folder exists
    If Not FileExists($folderPath) Then
        MsgBox(16, "Error", "The folder path does not exist: " & $folderPath)
        Return -1
    EndIf

    ; Look for PNG files
    Local $searchPattern = "*.png"
    Local $searchHandle = FileFindFirstFile($folderPath & "\" & $searchPattern)

    ; Loop through and find all .png files
    While 1
        Local $file = FileFindNextFile($searchHandle)
        If @error Then ExitLoop

        ; Skip files with "_drawbox" or "_inpainted" in the name
        If StringInStr($file, "_drawbox") = 0 And StringInStr($file, "_inpainted") = 0 Then
            _ArrayAdd($fileList, $file)
        EndIf
    WEnd

    FileClose($searchHandle)

    ; Look for JPG files
    $searchPattern = "*.jpg"
    $searchHandle = FileFindFirstFile($folderPath & "\" & $searchPattern)

    ; Loop through and find all .jpg files
    While 1
        Local $file = FileFindNextFile($searchHandle)
        If @error Then ExitLoop

        ; Skip files with "_drawbox" or "_inpainted" in the name
        If StringInStr($file, "_drawbox") = 0 And StringInStr($file, "_inpainted") = 0 Then
            _ArrayAdd($fileList, $file)
        EndIf
    WEnd

    FileClose($searchHandle)

    ; Remove the initial empty element if no files were found
    If $fileList[0] = "" And UBound($fileList) > 1 Then
        _ArrayDelete($fileList, 0)
    EndIf

    ; Check if no files were found
    If UBound($fileList) = 1 And $fileList[0] = "" Then
        MsgBox(16, "Error", "No valid image files found in the directory. [1]")
        Return -1
    EndIf

    Return $fileList
EndFunc

Func ProcessImages($baseFolderPath, $sleepTime, $fillMethod, $gptValCheckbox, $expandPixels)
    MsgBox(0, "Starting", "Starting the process with " & $fillMethod & "...")
    WinSetState("Automation", "", @SW_MINIMIZE)

    ; Activate the already opened Photoshop window
    If WinExists("[CLASS:Photoshop]") Then
        WinActivate("[CLASS:Photoshop]")
    Else
        MsgBox(16, "Error", "Photoshop is not running.")
        Exit
    EndIf

    ; Iterate through each image file
    For $imageIndex = 0 To UBound($imageFiles) - 1
        Local $imageFile = $imageFiles[$imageIndex]

        Local $imageName = StringReplace($imageFile, ".png", "_inpainted.png")
        SendAndLog("Image name: " & $imageName)
        $imageName = StringReplace($imageName, ".jpg", "_inpainted.png")
        $imageBaseName = StringSplit($imageFile, ".")
        ; Construct the JSON file name by replacing the image extension with "_data_inpaint.json"
        Local $jsonFileName = StringReplace($imageFile, ".png", "_data_inpaint.json")
        $jsonFileName = StringReplace($jsonFileName, ".jpg", "_data_inpaint.json")

        ; Path to the JSON file
        Local $jsonFilePath = $baseFolderPath & "\" & $jsonFileName

        ; Open JSON File and parse data
        Local $jsonData = FileRead($jsonFilePath)
        If @error Then
            MsgBox(16, "Error", "Could not read JSON file: " & $jsonFilePath)
            ContinueLoop
        EndIf

        Local $oJSON = _JSON_Parse($jsonData)
        If @error Then
            MsgBox(16, "Error", "Error parsing JSON data from: " & $jsonFilePath)
            ContinueLoop
        EndIf

        Local $dataArray = $oJSON.data

        Sleep(3000)
        Send("^o")
        Sleep(3000)
        Send($imageFile)
        Sleep(500)
        Send("{TAB}")
        Sleep(500)
        Send("{TAB}")
        Sleep(500)
        Send("{ENTER}")
        Sleep(3000)

        For $i = 0 To UBound($dataArray) - 1
            Local $entry = $dataArray[$i]

            ; Select all (Ctrl + A)
            Send("^a")
            Sleep(500)

            ; Transform selection (Select > Transform Selection)
            Send("!s") ; Alt + S for Select
            Send("t") ; T for Transform Selection
            Sleep(1000)

            ; Enter the values from JSON (X, Y, W, H)
            Local $centerX = $entry[0]
            Local $centerY = $entry[1]
            Local $width = $entry[2]
            Local $height = $entry[3]
            Local $angle = $entry[4]

            ; Move mouse to X field and enter value
            MouseMove($coordX1, $coordY1) ; Move to X field
            Sleep(1000)
            MouseClick("left")
            Sleep(500)
            Send("^a")
            Send($centerX & " px") ; Type the X value from JSON

            ; Move mouse to Y field and enter value
            Send("{TAB}") ; Move to Y field
            Sleep(200)
            Send($centerY & " px") ; Type the Y value from JSON

            ; Move mouse to W (Width) field and enter value
            Send("{TAB}") ; Move to W field
            Sleep(200)
            Send($width & " px") ; Type the Width value from JSON

            ; Move mouse to H (Height) field and enter value
            Send("{TAB}") ; Move to H field
            Sleep(200)
            Send($height & " px") ; Type the Height value from JSON

            ; Move mouse to Angle field and enter value
            Send("{TAB}") ; Move to Angle field
            Sleep(200)
            Send($angle) ; Type the Width value from JSON

            Sleep(500)

            ; Press Enter to apply the transformation
            Send("{ENTER}")
            Sleep(500)
            Send("{ENTER}")
            Sleep(500)

            ; Expand selection
            Send("!+^{F1}")
            Sleep(500)
            Send($expandPixels)
            Sleep(500)
            Send("{ENTER}")
            
            ; Select Object selection tool
            ;~ Send("w")
            ;~ Sleep(300)
            ;~ MouseMove($genFillCoordinates[$genFill_idx-1][0], $genFillCoordinates[$genFill_idx-1][1]) ; Move to X field
            ; Choose Fill Method
            If $fillMethod = "Generative Fill" Then
                Send("!+^g") ; Shortcut for Generative Fill
				Sleep(1000)
				Send("{TAB}")
				Sleep(1000)
				Send("{ENTER}")
				Sleep(2000)
				MouseMove($coordX2, $coordY2) ; Move Mouse
				Sleep(1000)
				MouseClick("left")
				Sleep(200)
				MouseClick("left")
            Else
                Send("+{F3}") ; Shortcut for Content-Aware Fill
            EndIf

            ; Wait for Photoshop to generate the results
            Sleep($sleepTime)
			
			; Save 3 results for generative fill
            If $fillMethod = "Generative Fill" And $gptValCheckbox Then
				For $genFill_idx = 1 To 3 
                    Sleep(300)
                    ;Select validation position
                    MouseMove($genFillCoordinates[$genFill_idx-1][0], $genFillCoordinates[$genFill_idx-1][1]) ; Move to X field
                    Sleep(500)
                    MouseClick("left")
                    Sleep(500)
                    ;~ SendAndLog($imageBaseName[1])
                    ;~ SendAndLog($genFill_idx)
                    ;~ SendAndLog($i+1)
                    ;Save image=
                    Local $imageNameSub = $imageBaseName[1] & "_genFill_" & ($i + 1) & "_" & $genFill_idx & ".png"

                    ;~ $imageNameSub = StringReplace($imageNameSub, ".jpg", "_" & $i+1 & ".png")
                    Send("!+^q")
                    Sleep(3000)
                    Send($imageNameSub)
                    Sleep(300)
                    Send("{TAB}")
                    Sleep(300)
                    Send("{TAB}")
                    Sleep(300)
                    Send("{TAB}")
                    Sleep(300)
                    Send("{ENTER}")
                    Send("{ENTER}")
                    Sleep(2000)
                Next
                Local $saveDir = $baseFolderPath & "\Generative fill"
                ;Check output folder
                If Not(FileExists($saveDir)) Then
                    DirCreate($saveDir)
                EndIf

                Local $imgOriBase64 = ConvertImageToBase64($baseFolderPath & "\" & $imageFile)
                Local $genFillBase64_1 = ConvertImageToBase64($saveDir & "\" & $imageBaseName[1] & "_genFill_" & ($i + 1) & "_" & 1 & ".png")
                Local $genFillBase64_2 = ConvertImageToBase64($saveDir & "\" & $imageBaseName[1] & "_genFill_" & ($i + 1) & "_" & 2 & ".png")
                Local $genFillBase64_3 = ConvertImageToBase64($saveDir & "\" & $imageBaseName[1] & "_genFill_" & ($i + 1) & "_" & 3 & ".png")
                ;~ ConsoleWrite($imgOriBase64)
                
                ;Ask GPT select best image
                Local $prompt_1 = "I am going to give 1 original image and 3 inpainted images with  [Center x coordinate, Center y coordinate, width, height]. Look carefully at the image and tell me which image is the best inpainted image with appropiate generated content and resolution. Give me the justification and then on a new line, you must give me the index of best image without add any words, Expected index output is 0, 1 or 2." 
                
                Local $prompt_coor = "Here is inpainted bounding box information: " & "[ " &  $centerX & ", " & $centerY & ", " & $width & ", " & $height &"]"
                Local $prompt_img = "Here are 1 original image and 3 inpainted images: "
                
                Local $message_tmp = '[{"type": "text", "text": "' & $prompt_1 & $prompt_coor & '"},' & _
                    '{"type": "image_url", "image_url": {"url": "data:image/png;base64,' & $imgOriBase64 & '"}},' & _
                    '{"type": "image_url", "image_url": {"url": "data:image/png;base64,' & $genFillBase64_1 & '"}},' & _
                    '{"type": "image_url", "image_url": {"url": "data:image/png;base64,' & $genFillBase64_2 & '"}},' & _
                    '{"type": "image_url", "image_url": {"url": "data:image/png;base64,' & $genFillBase64_3 & '"}}]'
                
                Local $best_genFill = ChatGPT_Request($message_tmp)
                If Not(StringInStr("|1|2|0|", "|" & $best_genFill & "|") > 0) Then
                    $best_genFill = ChatGPT_Request($message_tmp)
                EndIf
                $best_genFill = Number($best_genFill)
                SendAndLog("Best option: " & $best_genFill & " - " & $imageBaseName[1] & "_genFill_" & ($i + 1) & "_" & ($best_genFill+1) & ".png")
                ;Select best position
                MouseMove($genFillCoordinates[$best_genFill][0], $genFillCoordinates[$best_genFill][1]) ; Move to X field
                Sleep(500)
                MouseClick("left")
                Sleep(500)


			ElseIf $saveSteps And Not($gptValCheckbox) Then
				Local $imageNameSub = StringReplace($imageName, ".png", "_" & $i+1 & ".png")
				$imageNameSub = StringReplace($imageNameSub, ".jpg", "_" & $i+1 & ".png")
				Send("!+^q")
				Sleep(3000)
				Send($imageNameSub)
				Sleep(500)
				Send("{TAB}")
				Sleep(500)
				Send("{TAB}")
				Sleep(500)
				Send("{TAB}")
				Sleep(500)
				Send("{ENTER}")
				Send("{ENTER}")
				Sleep(3000)
			EndIf
        Next

        Send("!+^q")
        Sleep(5000)
        Send($imageName)
        Sleep(500)
        Send("{TAB}")
        Sleep(500)
        Send("{TAB}")
        Sleep(500)
        Send("{TAB}")
        Sleep(500)
        Send("{ENTER}")
        Send("{ENTER}")
        Sleep(5000)
        Send("^w")
        Sleep(2000)
        Send("n")
        Sleep(1000)
    Next

    MsgBox(0, "Finished", "All images processed.")
EndFunc
