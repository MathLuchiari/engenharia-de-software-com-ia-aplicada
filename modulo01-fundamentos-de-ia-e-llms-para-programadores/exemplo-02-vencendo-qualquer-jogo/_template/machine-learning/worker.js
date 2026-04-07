importScripts('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest');

const MODEL_PATH = `yolov5n_web_model/model.json`;
const LABELS_PATH = `yolov5n_web_model/labels.json`;
const INPUT_MODEL_DIMENTIONS = 640;
const CLASS_THRESHOLD = 0.4;

let _labels = {}
let _model = null
async function loadModelAndLabels() {
    await tf.ready();

    _labels =  await( await fetch(LABELS_PATH)).json();
    _model = await tf.loadGraphModel(MODEL_PATH);

    //warmup
    const dummyInput = tf.ones(_model.inputs[0].shape);
    await _model.executeAsync(dummyInput);
    tf.dispose(dummyInput);

    postMessage({type: 'model-load'})
}

/**
    Pré-processa a imagem para o formato aceito pelo YOLO:
    - tf.browser.fromPixels(): converte ImageBitmap/ImageData para tensor [H, W, 3]
    - tf.image.resizeBilinear(): redimensiona para [INPUT_DIM, INPUT_DIM]
    - .div(255): Normaliza os valires para[0, 1]
    - .expandDims(0): Adiciona dimensão batch [1, H, W, 3]

    Uso de tf.tidy():
    - Garante que tensores temporários serão descartados automaticamente evitando vazamento de memória.
 */
function preprocessImage( input ) {
    return tf.tidy(() => {
        const image = tf.browser.fromPixels(input)

        return tf.image
                .resizeBilinear(image, [INPUT_MODEL_DIMENTIONS, INPUT_MODEL_DIMENTIONS])
                .div(255)
                .expandDims(0);
    }); 
}

async function runInference(tensor) {
    const output = await _model.executeAsync(tensor);
    tf.dispose(tensor);

    const [boxes, scores, classes] = output.slice(0, 3)
    const [boxesData, scoresData, classesData] = await Promise.all(
        [
            boxes.data(), 
            scores.data(), 
            classes.data()
        ]
    )

    output.forEach(t => t.dispose())

    return {
        boxes: boxesData, 
        scores: scoresData, 
        classes: classesData
    }
}

/**
 * Filtra e processa as predições:
 * - Aplica o lomoar de confiança (CLASS_THRESHOLD) para descartar predições fracas.
 * - Filtra apenas a classe desejada (ex: 'kite').
 * - Converte coordenadas normalizadas para pixels reais.
 * - Calcula o centro do bounding box para o clique do jogo.
 * 
 * Uso de generator function:
 * - Permite iterar sobre as predições filtradas de forma eficiente sem criar arrays intermediários.
 */
function * processPrediction({boxes, scores, classes}, width, height) {
    for(let index=0; index < scores.length; index++) {
        if(scores[index] < CLASS_THRESHOLD) continue;

        const label = _labels[classes[index]]
        if(label !== 'kite') continue;

        let [x1, y1, x2, y2] = boxes.slice(index * 4, (index + 1) * 4)
        x1 *= width
        y1 *= height
        x2 *= width
        y2 *= height
        
        const boxWidth = x2 - x1
        const boxHeight = y2 - y1
        const centerX = x1 + boxWidth / 2
        const centerY = y1 + boxHeight / 2

        yield { x: centerX, y: centerY, score: (scores[index] * 100).toFixed(2) }
    }
}

loadModelAndLabels()

self.onmessage = async ({ data }) => {
    if (data.type !== 'predict') return
    if(!_model) return;

    const input = preprocessImage(data.image)
    const { width, height } = data.image

    const inferenceResults = await runInference(input)

    for(const prediction of processPrediction(inferenceResults, width, height)) {
        postMessage({
            type: 'prediction',
            x: prediction.x,
            y: prediction.y,
            score: prediction.score
        });
    }
};

console.log('🧠 YOLOv5n Web Worker initialized');
