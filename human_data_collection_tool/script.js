document.addEventListener('DOMContentLoaded', () => {
    // configuration
    const TXT_PATH = 'human_baseline_test_subset/descr.txt';
    const AUDIO_PATH = 'audio/tick.wav';
    const QUESTION_TIME_LIMIT_S = 8;

    const POSSIBLE_ANSWERS = [
        "square", "circle", "bottom", "right", "top", "left",
        "1", "2", "3", "4", "5", "6",
        "green", "yellow", "pink", "red", "blue", "orange"
    ];
    const SUBTYPE_ANSWERS = {
        closest: ["pink", "red", "blue", "green", "yellow", "orange"],
        farthest: ["square", "circle"],
        count: ["1", "2", "3", "4", "5", "6"],
        shape: ["square", "circle"],
        topbottom: ["top", "bottom"],
        leftright: ["left", "right"]
    };
    // plot categories and colors
    const PLOT_CATEGORIES = ['shape', 'topbottom', 'leftright', 'closest', 'farthest', 'count', 'non-relational', 'relational', 'overall'];
    const PLOT_COLORS = {
        'shape': 'rgb(40, 167, 69)',    // green
        'topbottom': 'rgb(40, 167, 69)',
        'leftright': 'rgb(40, 167, 69)',
        'closest': 'rgb(40, 110, 180)',  // blue
        'farthest': 'rgb(40, 110, 180)',
        'count': 'rgb(40, 110, 180)',
        'non-relational': 'rgb(253, 126, 20)', // orange
        'relational': 'rgb(253, 126, 20)',
        'overall': 'rgb(220, 53, 69)'      // red
    };

    const introContainer = document.getElementById('intro-container');
    const startButton = document.getElementById('start-button');
    const exampleImageEl = document.getElementById('example-image');
    const loadingErrorEl = document.getElementById('loading-error');
    const testContainer = document.getElementById('test-container');
    const statusDiv = document.getElementById('status');
    const timerSpan = document.getElementById('time');
    const timerContainer = document.getElementById('timer');
    const imageEl = document.getElementById('test-image');
    const imageContainer = document.getElementById('image-container');
    const questionContainer = document.getElementById('question-container');
    const questionTextEl = document.getElementById('question-text');
    const answersContainer = document.getElementById('answers-container');
    const resultsContainer = document.getElementById('results-container');
    const sessionIdDisplay = document.getElementById('session-id-display');
    const downloadLink = document.getElementById('download-link');
    const resultsChartCtx = document.getElementById('results-chart')?.getContext('2d');

    // state
    let allQuestions = [];
    let currentQuestionIndex = 0;
    let sessionResults = [];
    let timerInterval = null;
    let questionStartTime = null;
    let sessionStartTime = null;
    let sessionId = null;
    let answerButtons = [];
    let audioContext = null;
    let nextSoundBuffer = null;
    let resultsChartInstance = null;

    function initAudio() {
        try {
            window.AudioContext = window.AudioContext || window.webkitAudioContext;
            if (!window.AudioContext) {
                console.warn("Web Audio API is not supported."); return;
            }
            audioContext = new AudioContext();
            fetch(AUDIO_PATH)
                .then(response => {
                    if (!response.ok) throw new Error(`Audio load error: ${response.status}`);
                    return response.arrayBuffer();
                })
                .then(buffer => audioContext.decodeAudioData(buffer))
                .then(decoded => { nextSoundBuffer = decoded; console.log("Audio loaded."); })
                .catch(e => console.error("Audio init failed:", e));
        } catch (e) {
            console.error("AudioContext init error:", e);
        }
    }

    function playSound() {
        if (!audioContext || !nextSoundBuffer) return;
        if (audioContext.state === 'suspended') {
            audioContext.resume().catch(e => console.error("Audio resume error:", e));
        }
        if (audioContext.state !== 'running') return;
        try {
            const source = audioContext.createBufferSource();
            source.buffer = nextSoundBuffer;
            source.connect(audioContext.destination);
            source.start(0);
        } catch (e) {
            console.error("Audio play error:", e);
        }
    }

    function generateUUID() {
        return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
            const r = Math.random() * 16 | 0, v = c === 'x' ? r : (r & 0x3 | 0x8);
            return v.toString(16);
        });
    }

    function getQuestionSubtype(vector) {
        if (!vector || vector.length < 11) return null;
        const isRelational = vector[6] === 1;
        const map = isRelational
            ? { 8: "closest", 9: "farthest", 10: "count" }
            : { 8: "topbottom", 9: "leftright", 10: "shape" };
        for (const idx in map) {
            if (vector.length > idx && vector[idx] === 1) return map[idx];
        }
        return null;
    }


    function parseTextData(text) {
        allQuestions = [];
        const lines = text.split('\n');
        console.log(`Read ${lines.length} lines from description file.`);

        lines.forEach((line, index) => {
            const trimmedLine = line.trim();
            if (!trimmedLine) return;

            const parts = trimmedLine.split('\t');
            if (parts.length !== 4) {
                console.warn(`Skipping line ${index + 1}: Incorrect number of columns (${parts.length}). Expected 4.`);
                return;
            }

            const [imagePath, question, correctAnswer, vectorString] = parts;

            if (!/^[01]+$/.test(vectorString)) {
                 console.warn(`Skipping line ${index + 1}: Invalid characters in question vector string: ${vectorString}`);
                 return;
            }
            const question_vector = vectorString.split('').map(Number);

             if (!Array.isArray(question_vector) || question_vector.length < 11) {
                 console.warn(`Skipping line ${index + 1} for image ${imagePath}: Invalid or too short question vector (length ${question_vector.length}).`);
                 return;
             }

            allQuestions.push({
                imagePath: imagePath,
                question: question,
                correctAnswer: String(correctAnswer),
                question_vector: question_vector
            });
        });

        console.log(`Successfully parsed ${allQuestions.length} valid questions.`);
    }


    function loadInitialData() {
        startButton.disabled = true;
        startButton.textContent = "Loading Data...";
        loadingErrorEl.style.display = 'none';

        fetch(TXT_PATH)
            .then(response => {
                if (!response.ok) throw new Error(`HTTP ${response.status}. Could not load ${TXT_PATH}.`);
                return response.text();
            })
            .then(textData => {
                if (!textData || textData.trim().length === 0) throw new Error("Data file is empty.");
                parseTextData(textData);
                if (allQuestions.length === 0) {
                     throw new Error("No valid questions could be parsed from the data file.");
                }
                displayExampleImage();
                initAudio();
                startButton.disabled = false;
                startButton.textContent = "Start Test";
            })
            .catch(error => {
                console.error("Initial load error:", error);
                loadingErrorEl.textContent = `Error loading data: ${error.message}. Cannot start. Check console, file path (${TXT_PATH}), and file format.`;
                loadingErrorEl.style.display = 'block';
                startButton.textContent = "Error Loading";
            });
    }

    function displayExampleImage() {
        if (allQuestions && allQuestions.length > 0) {
            exampleImageEl.src = allQuestions[0].imagePath;
            exampleImageEl.alt = "Example visual reasoning image";
        } else {
             exampleImageEl.alt = "Could not load example image - no questions parsed";
             console.warn("No questions available to display an example image.");
        }
    }

    function initializeTest() {
        if (allQuestions.length === 0) {
            console.error("Cannot initialize test - no questions loaded.");
            statusDiv.textContent = 'Error: No questions available.';
            statusDiv.style.color = 'red';
             testContainer.style.display = 'none';
             introContainer.style.display = 'block';
             loadingErrorEl.textContent = 'Error: No valid questions found. Cannot start the test.';
             loadingErrorEl.style.display = 'block';
             startButton.disabled = true;
            return;
        }

        statusDiv.textContent = 'Preparing test...';
        statusDiv.style.color = '#666';

        createAnswerButtons();
        startSession();
    }


    function createAnswerButtons() {
        answersContainer.innerHTML = '';
        answerButtons = [];
        POSSIBLE_ANSWERS.forEach(answer => {
            const button = document.createElement('button');
            button.textContent = answer;
            button.dataset.answer = answer;
            button.style.display = 'none';
            button.addEventListener('click', handleAnswerClick);
            answersContainer.appendChild(button);
            answerButtons.push(button);
        });
    }

    function startSession() {
        if (allQuestions.length === 0) {
            console.error("Attempted to start session with no questions.");
            return;
        }
        sessionId = generateUUID();
        sessionStartTime = new Date().toISOString();
        currentQuestionIndex = 0;
        sessionResults = [];
        resultsContainer.style.display = 'none';
        if (resultsChartInstance) { resultsChartInstance.destroy(); resultsChartInstance = null; }
        statusDiv.style.display = 'block';
        timerContainer.style.display = 'block';
        imageContainer.style.display = 'flex';
        questionContainer.style.display = 'block';
        answersContainer.style.display = 'flex';
        displayQuestion();
    }

    function displayQuestion() {
        if (currentQuestionIndex >= allQuestions.length) {
            endSession(); return;
        }
        const q = allQuestions[currentQuestionIndex];
        statusDiv.textContent = `Question ${currentQuestionIndex + 1} of ${allQuestions.length}`;
        imageEl.src = q.imagePath;
        imageEl.alt = `Test Image ${currentQuestionIndex + 1}`;
        questionTextEl.textContent = q.question;

        const subtype = getQuestionSubtype(q.question_vector);
        const validAnswers = (subtype && SUBTYPE_ANSWERS[subtype]) ? SUBTYPE_ANSWERS[subtype] : POSSIBLE_ANSWERS;

        let visible = [];
        let leftBtn = null, rightBtn = null;

        answerButtons.forEach(btn => {
            const answer = btn.dataset.answer;
            if (validAnswers.includes(answer)) {
                btn.style.display = 'inline-block';
                visible.push(btn);
                if (answer === 'left') leftBtn = btn;
                if (answer === 'right') rightBtn = btn;
            } else {
                btn.style.display = 'none';
            }
        });

         if (subtype === 'leftright' && leftBtn && rightBtn) {
            const lIdx = visible.indexOf(leftBtn);
            const rIdx = visible.indexOf(rightBtn);
            if (rIdx !== -1 && lIdx !== -1 && rIdx < lIdx) {
                [visible[lIdx], visible[rIdx]] = [visible[rIdx], visible[lIdx]];
            }
        } else if (subtype === 'count') {
            visible.sort((a, b) => parseInt(a.dataset.answer) - parseInt(b.dataset.answer));
        }

        answersContainer.innerHTML = '';
        visible.forEach(btn => answersContainer.appendChild(btn));

        resetTimer();
        startTimer();
        questionStartTime = Date.now();
    }

    function handleAnswerClick(e) {
        recordResult(e.target.dataset.answer, false);
        moveToNextQuestion();
    }

    function handleTimeout() {
        recordResult(null, true);
        moveToNextQuestion();
    }

     function recordResult(userAnswer, timedOut) {
        if (currentQuestionIndex >= allQuestions.length) return;
        const q = allQuestions[currentQuestionIndex];
        const isCorrect = !timedOut && userAnswer === q.correctAnswer;

        sessionResults.push({
            questionIndex: currentQuestionIndex,
            imagePath: q.imagePath,
            question: q.question,
            questionVector: q.question_vector,
            subtype: getQuestionSubtype(q.question_vector),
            userAnswer: userAnswer,
            correctAnswer: q.correctAnswer,
            isCorrect: isCorrect,
            timeTakenMs: Date.now() - questionStartTime,
            timedOut: timedOut,
        });
     }


    function moveToNextQuestion() {
        stopTimer();
        playSound();
        currentQuestionIndex++;
        displayQuestion();
    }


    function startTimer() {
        let timeLeft = QUESTION_TIME_LIMIT_S;
        timerSpan.textContent = timeLeft;
        timerInterval = setInterval(() => {
            timeLeft--;
            timerSpan.textContent = timeLeft;
            if (timeLeft <= 0) {
                handleTimeout();
            }
        }, 1000);
    }

    function stopTimer() {
        clearInterval(timerInterval);
        timerInterval = null;
    }

    function resetTimer() {
        stopTimer();
        timerSpan.textContent = QUESTION_TIME_LIMIT_S;
    }


    function calculateAccuracies() {
        const totals = { overall: 0, relational: 0, 'non-relational': 0 };
        const corrects = { overall: 0, relational: 0, 'non-relational': 0 };
        PLOT_CATEGORIES.forEach(cat => {
             if (!totals[cat]) totals[cat] = 0;
             if (!corrects[cat]) corrects[cat] = 0;
        });


        sessionResults.forEach(res => {
            if (!res.questionVector || res.questionVector.length < 11) {
                console.warn("Skipping result in accuracy calculation due to invalid vector:", res);
                return;
            }

            const subtype = res.subtype;
            const isRelational = res.questionVector[6] === 1;
            const category = isRelational ? 'relational' : 'non-relational';

            totals.overall++;
            totals[category]++;
            if (subtype && totals[subtype] !== undefined) {
                 totals[subtype]++;
            }


            if (res.isCorrect) {
                corrects.overall++;
                corrects[category]++;
                 if (subtype && corrects[subtype] !== undefined) {
                     corrects[subtype]++;
                 }
            }
        });

        const accuracies = {};
        PLOT_CATEGORIES.forEach(cat => {
            accuracies[cat] = (totals[cat] !== undefined && totals[cat] > 0) ? (corrects[cat] / totals[cat]) : 0;
        });

        console.log("Calculated Accuracies:", accuracies);
        return accuracies;
    }

    function generateResultsChart(accuracies) {
        if (!resultsChartCtx) {
            console.error("Canvas context not found for chart.");
            return Promise.reject("Canvas context not found");
        }

         if (resultsChartInstance) {
            resultsChartInstance.destroy();
         }


        const chartData = PLOT_CATEGORIES.map(cat => accuracies[cat] !== undefined ? accuracies[cat] : 0);
        const backgroundColors = PLOT_CATEGORIES.map(cat => PLOT_COLORS[cat] || 'grey');

        return new Promise((resolve) => {
            resultsChartInstance = new Chart(resultsChartCtx, {
                type: 'bar',
                data: {
                    labels: PLOT_CATEGORIES,
                    datasets: [{
                        label: 'Accuracy',
                        data: chartData,
                        backgroundColor: backgroundColors,
                        borderColor: backgroundColors,
                        borderWidth: 1
                    }]
                },
                options: {
                    indexAxis: 'y',
                    scales: {
                        x: {
                            beginAtZero: true,
                            max: 1,
                            title: { display: true, text: 'Accuracy' },
                            ticks: { stepSize: 0.2 }
                        },
                        y: { beginAtZero: true }
                    },
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    let label = context.dataset.label || '';
                                    if (label) { label += ': '; }
                                    if (context.parsed.x !== null) {
                                        label += context.parsed.x.toFixed(4);
                                    }
                                    return label;
                                }
                            }
                        },
                         datalabels: {
                            anchor: 'end',
                            align: 'right',
                            formatter: (value) => (value * 100).toFixed(2) + '%',
                            color: '#333',
                            offset: 4,
                            font: { size: 10 }
                        }
                    },
                    animation: {
                        onComplete: () => {
                             console.log("Chart animation complete.");
                             resolve();
                        }
                    }
                },
            });
             setTimeout(resolve, 500);
        });
    }

    async function generateZip(finalOutput, chartInstance) {
         if (!chartInstance || !chartInstance.canvas) {
            console.error("Chart instance or canvas not available for ZIP generation.");
            downloadLink.textContent = "Error creating ZIP (chart missing)";
            downloadLink.style.display = 'inline-block';
            return;
         }
         if (typeof JSZip === 'undefined') {
            console.error("JSZip library not loaded.");
             downloadLink.textContent = "ZIP Library Error";
             downloadLink.style.display = 'inline-block';
            return;
         }

         const zip = new JSZip();

         const jsonString = JSON.stringify(finalOutput, null, 2);
         zip.file("results.json", jsonString);

         try {
            const chartImageDataUrl = chartInstance.toBase64Image('image/png');
            if (!chartImageDataUrl || chartImageDataUrl === 'data:,') {
                 throw new Error("Generated chart image data URL is empty or invalid.");
            }
            const base64Response = await fetch(chartImageDataUrl);
            const chartBlob = await base64Response.blob();
             zip.file("accuracy_plot.png", chartBlob, { binary: true });

             const zipBlob = await zip.generateAsync({ type: "blob" });

             const url = URL.createObjectURL(zipBlob);
             downloadLink.href = url;
             downloadLink.download = `${finalOutput.sessionId}_results.zip`;
             downloadLink.textContent = "Download Results ZIP";
             downloadLink.style.display = 'inline-block';

         } catch (error) {
            console.error("Error generating ZIP file:", error);
            downloadLink.textContent = "Error creating ZIP";
            downloadLink.style.display = 'inline-block';
         }
     }


    async function endSession() {
        stopTimer();
        const sessionEndTime = new Date().toISOString();
        const totalTimeMs = sessionStartTime ? (new Date(sessionEndTime).getTime() - new Date(sessionStartTime).getTime()) : 0;

        console.log("Session Ended. Calculating results...");
        testContainer.style.display = 'none';

        const accuracies = calculateAccuracies();

        const finalOutput = {
            sessionId: sessionId,
            sessionStartTime: sessionStartTime,
            sessionEndTime: sessionEndTime,
            totalTimeMs: totalTimeMs,
            totalQuestionsAttempted: sessionResults.length,
            totalQuestionsAvailable: allQuestions.length,
            accuracySummary: accuracies,
            results: sessionResults
        };

        resultsContainer.style.display = 'block';
        sessionIdDisplay.textContent = sessionId;
        downloadLink.textContent = "Generating Download...";
        downloadLink.style.display = 'inline-block';
        downloadLink.href="#";


        try {
            console.log("Generating results chart...");
            await generateResultsChart(accuracies);
            console.log("Chart generated, preparing ZIP...");
            if (resultsChartInstance) {
                 await generateZip(finalOutput, resultsChartInstance);
                 console.log("ZIP preparation complete.");
            } else {
                 console.error("Chart instance not available after generation.");
                 downloadLink.textContent = "Error creating ZIP (Chart failed)";
            }
        } catch(error) {
             console.error("Error during end session processing (chart/zip):", error);
             downloadLink.textContent = "Error creating Download";
        }

    }

    startButton.addEventListener('click', () => {
         if (audioContext && audioContext.state === 'suspended') {
            audioContext.resume().catch(e => console.error("Audio resume error on start:", e));
         }
        introContainer.style.display = 'none';
        testContainer.style.display = 'block';
        initializeTest();
    });

    loadInitialData();

});