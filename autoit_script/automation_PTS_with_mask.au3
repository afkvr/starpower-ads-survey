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
Global $imageMaskFiles


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
            ConsoleWrite($imageFiles & @CRLF)
            Local $fillMethod = "Generative Fill"
            If GUICtrlRead($contentAwareFill_Radio) = $GUI_CHECKED Then
                $fillMethod = "Content-Aware Fill"
            EndIf

            If UBound($imageFiles) > 0 Then
                $imageMaskFiles = GetMaskFiles($baseFolderPath, $imageFiles)
                ProcessImages_Mask($baseFolderPath, $sleepTime, $fillMethod, $gptValCheckbox, $expandPixels)
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
        If StringInStr($file, "_mask") = 0 And StringInStr($file, "_trans") = 0 And StringInStr($file, "_inpainted") = 0 Then
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
        If StringInStr($file, "_mask") = 0 And StringInStr($file, "_trans") = 0 And StringInStr($file, "_inpainted") = 0 Then
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


Func GetMaskFiles($folderPath, $imageFiles)
    Local $maskMaps[UBound($imageFiles)][1]
    ; Scan mask file according scanned image file
    For $imageIndex = 0 To UBound($imageFiles) - 1
        Local $searchPattern = "*mask*"
        Local $searchHandle = FileFindFirstFile($folderPath & "\" & $searchPattern)
        Local $imageFile = $imageFiles[$imageIndex]
        Local $maskList[1]
        $maskList[0] = ""
        $imageBaseName = StringSplit($imageFile, ".")
        ; Loop through and find mask file has image basename inside
        While 1
            Local $file = FileFindNextFile($searchHandle)
            If @error Then ExitLoop
            ; Skip files with "_drawbox" or "_inpainted" in the name
            If StringInStr($file,$imageBaseName[1]) = 1 Then
                _ArrayAdd($maskList, $file)
            EndIf
        WEnd
        
        ; Remove the initial empty element if no files were found
        If $maskList[0] = "" And UBound($maskList) > 1 Then
            _ArrayDelete($maskList, 0)
        EndIf
        $maskMaps[$imageIndex][0] = $maskList
        ;~ _ArrayDisplay($maskList, $imageFiles[$imageIndex])
        ;~ _ArrayDisplay($maskMaps[0][0], "Internal array")
        FileClose($searchHandle)
    Next 

    Return $maskMaps
EndFunc



Func ProcessImages_Mask($baseFolderPath, $sleepTime, $fillMethod, $gptValCheckbox, $expandPixels)
    MsgBox(0, "Starting", "Starting the process with " & $fillMethod & "...")
    WinSetState("Automation", "", @SW_MINIMIZE)

    ; Activate the already opened Photoshop window
    If WinExists("[CLASS:Photoshop]") Then
        WinActivate("[CLASS:Photoshop]")
    Else
        MsgBox(16, "Error", "Photoshop is not running.")
        Exit
    EndIf

    Local $saveDir = $baseFolderPath & "\Generative fill"
    ;Check output folder
    If Not(FileExists($saveDir)) Then
        DirCreate($saveDir)
    EndIf

    ; Iterate through each image file
    For $imageIndex = 0 To UBound($imageFiles) - 1
        Local $imageFile = $imageFiles[$imageIndex]

        $imageBaseName = StringSplit($imageFile, ".")
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
        Local $maskFiles = $imageMaskFiles[$imageIndex][0]
        Sleep(1000)
        For $maskIndex = 0 To UBound($maskFiles) - 1
            ;~ Local $maskName = StringReplace($imageFile, ".jpg", "_mask_" & $maskIndex+1 &".jpg")
            Sleep(500)
            Local $maskName = $maskFiles[$maskIndex]
            ; Working instructions for running the action script 
            
            If $fillMethod = "Generative Fill" Then
                
                ;Handle PTS warning funtion "Make" from selected mask - Cause by previous genFill mask layer.
                If $maskIndex > 0 Then 
                    Sleep(1000)
                    Send("+{F9}")  ;Shift+F9 to run action delete current mask layer
                EndIf

                ; Start action "Generate fill"
                Sleep(1000)
                Send("+{F10}")  ;Shift+F10 to run action genfill + masks + Expand custom
                Sleep(1000)
                ;~ Fill input image mask name
                Send("{TAB}")
                Sleep(500)
                Send($maskName)
                Sleep(2000)
                Send("{TAB}")
                Sleep(500)
                Send("{TAB}")
                Sleep(500)
                Send("{TAB}")
                Sleep(500)
                Send("{ENTER}")
    
                ;~ Confirm place layer
                Sleep(2000)
                Send("{ENTER}")
                
                ;~ Confirm color range
                Sleep(2000)
                Send("{ENTER}")
                Sleep(2000)
                Send("{ENTER}")

                ;~ Fullfill expand selection and confirm to load selection
                Sleep(3000)
                Send($expandPixels)
                Sleep(2000)
                Send("{ENTER}")
                Sleep(2000)
                Send("{ENTER}")

                ; Pass empty prompt and gen Fill
                Sleep(3000)
                Send("{TAB}")
                Sleep(1000)
                Send("{ENTER}")
                Sleep($sleepTime)

            Else
                Send("+{F3}") ; Shortcut for Content-Aware Fill
            EndIf

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
                    Local $imageNameSub = $saveDir & "\" & $imageBaseName[1] & "_genFill_" & ($maskIndex + 1) & "_" & $genFill_idx & ".png"

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
                    ;~ Send("{ENTER}")
                    Sleep(2000)
                Next

                Local $imgOriBase64 = ConvertImageToBase64($baseFolderPath & "\" & $imageFile)
                Local $maskBase64 = ConvertImageToBase64($baseFolderPath & "\" & $maskName)
                Local $genFillBase64_1 = ConvertImageToBase64($saveDir & "\" & $imageBaseName[1] & "_genFill_" & ($maskIndex + 1) & "_" & 1 & ".png")
                Local $genFillBase64_2 = ConvertImageToBase64($saveDir & "\" & $imageBaseName[1] & "_genFill_" & ($maskIndex + 1) & "_" & 2 & ".png")
                Local $genFillBase64_3 = ConvertImageToBase64($saveDir & "\" & $imageBaseName[1] & "_genFill_" & ($maskIndex + 1) & "_" & 3 & ".png")
                ;~ ConsoleWrite($imgOriBase64)
                
                ;Ask GPT select best image
                
         		Local $prompt_1 = "I will provide 1 original image, 1 mask image, and 3 inpainted images where all objects have been removed from the original image using the mask. Your task is to carefully evaluate the 3 inpainted images and determine which one has the most appropriate generated content and resolution, ensuring it blends naturally with the original image. Provide a justification for your choice and, on a new line, give the index of the best image (0, 1, or 2) without any additional words or explanation." 
 			
                ;~ Local $prompt_coor = "Here is inpainted bounding box information: " & "[ " &  $centerX & ", " & $centerY & ", " & $width & ", " & $height &"]"
                
                Local $message_tmp = '[{"type": "text", "text": "' & $prompt_1 & '"},' & _
                    '{"type": "image_url", "image_url": {"url": "data:image/png;base64,' & $imgOriBase64 & '"}},' & _
                    '{"type": "image_url", "image_url": {"url": "data:image/png;base64,' & $maskBase64 & '"}},' & _
                    '{"type": "image_url", "image_url": {"url": "data:image/png;base64,' & $genFillBase64_1 & '"}},' & _
                    '{"type": "image_url", "image_url": {"url": "data:image/png;base64,' & $genFillBase64_2 & '"}},' & _
                    '{"type": "image_url", "image_url": {"url": "data:image/png;base64,' & $genFillBase64_3 & '"}}]'
                
                Local $best_genFill = ChatGPT_Request($message_tmp)
                If Not(StringInStr("|1|2|0|", "|" & $best_genFill & "|") > 0) Then
                    $best_genFill = ChatGPT_Request($message_tmp)
                EndIf
                $best_genFill = Number($best_genFill)
                SendAndLog("Best option: " & $best_genFill & " - " & $imageBaseName[1] & "_genFill_" & ($maskIndex + 1) & "_" & ($best_genFill+1) & ".png")
                ;Select best position
                MouseMove($genFillCoordinates[$best_genFill][0], $genFillCoordinates[$best_genFill][1]) ; Move to X field
                Sleep(500)
                MouseClick("left")
                Sleep(500)

            ElseIf $saveSteps And Not($gptValCheckbox) Then
                Local $imageNameSub = StringReplace($imageName, ".png", "_" & $maskIndex+1 & ".png")
                $imageNameSub = StringReplace($imageNameSub, ".jpg", "_" & $maskIndex+1 & ".png")
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
        Local $imageSavePath= $saveDir & "\" & $imageBaseName[1] & "_inpainted.png"
        Sleep(5000)
        
        Send($imageSavePath)
        Sleep(500)
        Send("{TAB}")
        Sleep(500)
        Send("{TAB}")
        Sleep(500)
        Send("{TAB}")
        Sleep(500)
        Send("{ENTER}")
        Sleep(5000)
        Send("^w")
        Sleep(2000)
        Send("n")
        Sleep(1000)
    Next

    MsgBox(0, "Finished", "All images processed.")
EndFunc
