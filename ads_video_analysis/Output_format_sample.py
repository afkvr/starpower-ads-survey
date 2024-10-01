### Font Recognition format:
font_recognition_output = {
    "data":[
        {
            "video-infos":{                                                # 1: For each video
                "video-name":"AA",
                "resolution":[100,100],                                    # 1-a: Video resolution in format [weight, height]
                "duration":0.0,                                            # 1-b: Video Duration (seconds)
                "num-text-blocks":1,                                       # 1-c Amount of Text Blocks in Video
                "ratio-text-duration":0.1,                                 # 1-d % of video with any text on screen (by duration)
                "ratio-text-area":0.1                                      # 1-e: % of video covered by text (% of video frame across duration)| fomula = (all_box_text_area x text_duration) / (frame_area x video_duration)
            },
            "sections-data":[                                              # 2: For each section
                {
                    "section-name":"section-1",                            # 2-a-i Section Name
                    "start-time":0,                                        # 2-a-ii Section Time Start (second)
                    "end-time":10,                                         # 2-a-iii Section Time End  (second)
                    "duration":10.0,                                       # 2-a-iv Section Duration Time (second)
                    "block-still-active":True,                             # 2-b-i Text box still active from previous section? (True/False)
                    "num-text-blocks":1,                                   # 2-b-ii Amount of Text Blocks in Section 
                    "ratio-text-duration":0.1,                             # 2-b-iii % of video with any text on screen (by duration) during this section
                    "ratio-text-area":0.1                                  # 2-b-iv % of video covered by text (% of video frame across duration) during this section
                },
                {
                    "section-name":"section-2",
                    "start-time":10,
                    "end-time":20,
                    "duration":10.0,
                    "num-text-blocks":1,
                    "block-still-active":True,
                    "ratio-text-duration":0.1,
                    "ratio-text-area":0.1
                }
            ],
            "textblocks-data":[                                            # 3 For each text block in the video
                {   
                    "block-name":"block-1",
                    "timing-info":{
                        "start-time":1.0,                                  # 3-a-i Time the text is first shown (second)
                        "end-time":3.0,                                    # 3-a-ii Time the text is last shown (second)
                        "duration":0.0,                                    # 3-a-iii Duration time from entry to exit (second)
                        "ratio-text-duration":0.1                          # 3-a-iv Duration of text as % of video length
                    },
                    "text-info":{
                        "text-content":"Aa aa",                            # 3-b-i Text capitalized and punctuated as shown
                        "wpm":1.2                                          # 3-a-ii WPM (number of words shown divided by duration time from entry to exit)
                    },
                    "block-size":[100,100],                                # 3-c: Size (objective), rectangle -  Format [weight, height] of box text
                    "ratio-block-frame":10.0,                              # 3-d: Size (% of video frame)- Fomula - box_text_area / frame_area
                    "block-coordinates":[[0,0],[0,50],[50,50],[50,0]],     # 3-e: Location (objective) - Save coordinate of box text like [[top-left], [top-right], [bottom-right],[bottom-left]]  
                    "block-sectors":[1,2,3,4,5],                            # 3-f: Location (section of video frame) (map to be included)
                    "font-details":{
                        "font-style":"https://www.whatfontis.com/NFS_Proxima-Nova-Semibold.font",  # 3-g-i: Font style (closest match to font) (Link font returned by whatfontis)
                        "font-size":20,                                    # 3-g-ii: Font size (in pt)
                        "serif":True,                                      # 3-g-iii: True is Serif | False is Sans Serif
                        "font-weight":"bold",                              # 3-g-iv: Font weight (bolded, unbolded)
                        "cap-style":"standard case",                       # 3-g-v: Capitalization style - output is 1 in 4 types: "standard case"/"Start Case"/"ALL CAPS"/"no caps"
                        "font-color":"#001400",                            # 3-g-vi: Font color (save as format HEX)
                        "font-outline":{                                   # 3-g-vii: Font outline (yes or no). If no => result as None/null. 
                            "color":"#001400",                             # If yes => output would has field "color"
                        },
                        "font-outline-weight":3                            # 3-g-viii: font outline weight. If "font-outline" is None, font-outline-weight equal 0.
                    },
                    "text-natural":True                                    # 3-h: Is the text natural in video or added on top? True is natural, False is added on top
                }
            ]
        }
    ]
}

### Speaker Recognition
speaker_recognition_output = {
    "data":[
        {
            "video-infos":{                                                # 1: For each video
                "video-name":"AA",                                         
                "resolution":[100,100],                                    # 1-a: Video resolution in format [weight, height]
                "duration":0.0,                                            # 1-b: Video Duration (seconds)
                "num-speakers":1,                                          # 1-c: Amount of Speakers in Video
                "speaker-infos":[
                    {
                        "speakerID":"face_1",                              # 1-c-i: Each speaker’s ID 
                        "gender":"Male",                                   # 1-c-ii: Gender (Male/Female)
                        "age-range":"(0-2)",                               # 1-c-iii: Est. age range
                        "ratio-speaker-duration":0.1,                      # 1-c-iv: % of video covered by each speaker ID (by duration)
                        "ratio-speaker-area":0.1                           # 1-c-v: % of video covered by each speaker ID (% of video frame across duration)
                    }
                ],
                "ratio-allspeakers-areas":0.1                                  # 1-d: % of video covered by speakers (% of video frame across duration)
            },
            "sections-data":[                                              # 2: For each section
                {
                    "section-name":"section-1",                                
                    "duration":5.0,                                        # 2-a: Section Duration (second)
                    "num-speakers":1,                                      # 2-b: Amount of Speakers in Section 
                    "speaker-infos":[
                        {
                            "speakerID":"1",                               # 2-b-i: Each speaker’s ID
                            "gender":"Male/Female",                        # 2-b-ii: Gender (Male/Female)
                            "age-range":"(0-2)",                           # 2-b-iii: Est. age range
                            "ratio-speaker-duration":0.1,                  # 2-b-iv: % of video covered by each speaker ID (by duration)
                            "ratio-speaker-area":0.1                       # 2-b-v: % of video covered by each speaker ID (% of video frame across duration)
                        }
                    ],
                    "ratio-allspeakers-areas":0.1                          # 2-c: % of Section covered by speakers (% of video frame across duration)
                },
                {
                    "section-name":"section-2",
                    "duration":5.0,
                    "num-speakers":1,
                    "speaker-infos":[
                        {
                            "speakerID":"face_1",
                            "gender":"Male/Female",
                            "age-range":"(0-2)",
                            "ratio-speaker-duration":0.1,
                            "ratio-speaker-area":0.1
                        }
                    ],
                    "ratio-allspeakers-areas":0.1
                }
            ],
            "speakers-data":[                                              # 3: For each speaking segment
                {
                    "speaker-info":{
                        "speakerID":"face_1",                              # 3-a-i: Speaker ID (if the same speaker is shown multiple times, should be the same ID)
                        "gender":"Male/Female",                            # 3-a-ii: Gender (Male/Female)
                        "age-range":"(0-2)",                               # 3-a-iii: Est. age range
                        "speaking":True                                    # 3-a-iv: Speaking? True is speaking, False is non-speaking
                    },
                    "timing-info":{
                        "start-time":0.0,                                  # Time the speaker is first shown
                        "end-time":10.0,                                   # Time the speaker is last shown
                        "duration":10.0,                                   # Duration time from entry to exit
                        "ratio-speaker-duration":0.1                       # Duration of speaker on video as % of video length
                    },
                    "block-size":[100,100],                                # Size (objective), rectangle
                    "ratio-block-frame":10.0,                              # Size (% of video frame)
                    "face-coordinates":[[0,0],[0,50],[50,50],[50,0]],      # 3-e: Location (objective) - Save coordinate of box text like [[top-left], [top-right], [bottom-right],[bottom-left]]  
                    "face-frames":[1,2,3,4,5],                             # 3-f: Location (section of video frame) (map to be included)
                    "speaker-show":"shoulderup"                            # 3-g: What is shown of speaker? => 1 in 3 options "shoulderup/waistup/fullbody"
                }
            ]
        }
    ]
}