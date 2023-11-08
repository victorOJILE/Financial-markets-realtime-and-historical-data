const html = document.documentElement;

function openFullScreen() {
	if (html.requestFullscreen) {
		html.requestFullscreen();
	} else if (html.webkitRequestFullscreen) {
		html.webkitRequestFullscreen();
	} else if (html.msRequestFullscreen) {
		html.msRequestFullscreen();
	}
}

function closeFullScreen() {
	if (html.exitFullscreen) {
		html.exitFullscreen();
	} else if (html.webkitExitFullscreen) {
		html.webkitExitFullscreen();
	} else if (html.msExitFullscreen) {
		html.msExitFullscreen();
	}
}

document.getElementById('expand').addEventListener('click', openFullScreen);
document.getElementById('compress').addEventListener('click', closeFullScreen);

function cEl(elem, props = {}, ...children) {
	let element = document.createElement(elem);
	if (props && typeof props == 'object') {
		for (let prop in props) {
			if (prop == 'data') {
				for (let d in props[prop]) {
					element.dataset[d] = props[prop][d];
				}
			} else {
				element[prop] = props[prop];
			}
		}
	}
	if (children) {
		for (let child of children) element.append(child);
	}
	return element;
}

iframe = iframe.contentDocument;
let mainElem = iframe.body;

/* toggle timeframe buttons background color */

let timeframeToggle = {
	"5M": 'TIME_SERIES_INTRADAY&symbol=_Quote_&interval=5min&outputsize=compact&apikey=demo',
	"15": 'TIME_SERIES_INTRADAY&symbol=_Quote_&interval=15min&outputsize=compact&apikey=demo',
	"30": 'TIME_SERIES_INTRADAY&symbol=_Quote_&interval=30min&outputsize=compact&apikey=demo',
	"1H": 'TIME_SERIES_INTRADAY&symbol=_Quote_&interval=60min&outputsize=compact&apikey=demo',
	"D1": 'TIME_SERIES_DAILY&symbol=_Quote_&outputsize=compact&apikey=demo',
	"W1": 'TIME_SERIES_WEEKLY&symbol=_Quote_&apikey=demo',
	getUrl(tf) {
	 let string = this[tf];
	 return string.replace('_Quote_', 'IBM');
	}
};

let visible = document.getElementsByClassName("visible")[0];
let tfBtns = visible.children;
let spinner = document.getElementsByClassName('fa-spin')[0];

tfBtns[1].classList.add('active');

let fetching = false;

visible.addEventListener('click', function(event) {
	if (fetching) return;
	let target = event.target;
	for (let key of tfBtns)	key.classList.remove('active');
	
	target.classList.add('active');
	
	tf.innerText = target.innerText;
	fetching = true;
	spinner.classList.remove('hide');
	fetchData(timeframeToggle.getUrl([target.textContent]));
});

const chartDiv = mainElem.firstElementChild;

const domHigh = document.getElementById('highest-price');
const domLow = document.getElementById('lowest-price');
const domOpen = document.getElementById('open-price');
const domClose = document.getElementById('current-price');

const canvas = iframe.getElementById('canvas');
const ctx = canvas.getContext('2d');
let candlesX = [];
const candleWidth = 5;
const candleSpacing = 8;

let width = canvas.width;
let height = canvas.height;
let type = localStorage.getItem('tpe') || 'line';
let gridColor = 'darkslategray';

let priceArray, px, high, low, len;

const grid = new class {
 constructor() {
	 this.saved = document.createElement('canvas');
	 this.saved.width = width;
	 this.saved.height = height;
	 this.ctx = this.saved.getContext('2d');
	 
	 const patternCanvas = document.createElement("canvas");
	 const patternCtx = patternCanvas.getContext("2d");
	 
	 patternCanvas.width = 30;
	 patternCanvas.height = 30;
	 
	 patternCtx.setLineDash([3, 8]);
	 patternCtx.strokeWidth = 2;
	 patternCtx.strokeStyle = gridColor;
	 patternCtx.moveTo(0, 2);
	 patternCtx.lineTo(width, 0);
	 
	 patternCtx.moveTo(0, 0);
	 patternCtx.lineTo(0, height);
	 
	 patternCtx.stroke();
	 
	 this.pattern = this.ctx.createPattern(patternCanvas, "repeat");
	 
	 this.drawGrid();
 }
	drawGrid() {
		this.ctx.fillStyle = this.pattern;
		this.ctx.fillRect(0, 0, width, height);
	}
	removeGrid() {
		this.ctx.clearRect(0, 0, width, height);
	}
}

function drawLine(x1, y1, x2, y2, c = ctx) {
	c.moveTo(x1, y1);
	c.lineTo(x2, y2);
	c.stroke();
}

class CandleSticks {
	constructor() {
		this.bullColor = 'white';
		this.bearColor = 'red';
		this.chartImg = document.createElement('canvas');
		this.chartImg.width = width;
		this.chartImg.height = height;
		this.chartImgCtx = this.chartImg.getContext('2d');
	}
	drawCandles() {
		let y1, y2;
		px = ((height / 1.2) / (high - low));
		let x1 = 80;
		let x2 = 80;
		candlesX = [];
	 
		ctx.clearRect(0, 0, width, height);
		ctx.drawImage(grid.saved, 0, 0);
		
		/* Draw a white line over the below black container */
		ctx.beginPath();
		ctx.strokeStyle = "white";
		drawLine(0, height-25, width, height-25);
		/* Draw a black container to cover grids and show time clearly */
		ctx.fillStyle = 'black';
		ctx.fillRect(0, height - 25, width, 25);
		ctx.fill();

		ctx.fillStyle = 'white';
		ctx.textAlign = 'center';
		
		function drawText(x, time) {
			ctx.strokeStyle = 'white';
			ctx.fillStyle = 'white';
			drawLine(x, height-25, x, height-21);
			ctx.fillText(time, x, height -12);
			ctx.restore();
		}
		
		let pSFLC = {
			x: x1, y: (high - priceArray[0].close) *px
		};
		
		for(let candle of priceArray) {
			if (type == 'line') {
				ctx.beginPath();
				ctx.lineWidth = 2;
				ctx.strokeStyle = this.bullColor;
				ctx.moveTo(pSFLC.x, pSFLC.y);
				let y = (high - candle.close) * px;
				ctx.lineTo(x1, y);
				ctx.fillStyle = this.bullColor;
				ctx.arc(x1, y, 2, 0, 2 * Math.PI);
				ctx.stroke();
				ctx.fill();
				
				pSFLC.x = x1;
				pSFLC.y = y;
				x1 += candleSpacing;
			} else {
				let info = this.getInfo(candle);

				y1 = info.type == 'bull' ? (high - candle.close) * px: (high - candle.open) * px;
				y2 = info.type == 'bull' ? ((high - candle.open) * px) - y1 : ((high - candle.close) * px) - y1;

				this._color = info.color;
				this.candleLine = (info.type == 'bull' ? this.bullCandleLineColor : this.bearCandleLineColor) || this._color;
				this.candlebody = this._color;
				
				ctx.beginPath();
				ctx.lineWidth = 1;
				ctx.strokeStyle = this.candleLine;
				drawLine(x1 + (candleWidth / 2), (high - candle.high) * px, x1 + (candleWidth / 2), (high - candle.low) * px);
				ctx.fillStyle = this.candlebody;
				ctx.fillRect(x2, y1, 5, y2);
				
				x1 += candleSpacing;
				x2 += candleSpacing;
			}
			
			if (candle.time == '00:00') { // Draw period seperator
				ctx.save();
				ctx.setLineDash([5, 3]);
				ctx.strokeStyle = 'white';
				drawLine(x1 + (type != 'line' ? (candleWidth /2) : 0), 0, x1 + (type != 'line' ? (candleWidth /2) : 0), height-25);
				drawText(x1 + (type != 'line' ? (candleWidth /2) : 0), '00:00');
			} else if (candle.time == '08:30') {
				ctx.save();
				drawText(x1 + (type != 'line' ? (candleWidth /2) : 0), '08:30');
			} else if (candle.time == '16:30') {
				ctx.save();
				drawText(x1 + (type != 'line' ? (candleWidth /2) : 0), '16:30');
			}
			candlesX.push(x1 + (type != 'line' ? (candleWidth /2) : 0)); // To be used for tracking x positions on chart
		}

		/* Draw bid line */
		ctx.beginPath();
		ctx.strokeStyle = "rgb(30, 30, 255)";
		drawLine(0, (high - priceArray[len].close) * px, width, (high - priceArray[len].close) * px);
		
		// Save current chart
		this.chartImgCtx.clearRect(0, 0, width, height);
		this.chartImgCtx.drawImage(canvas, 0, 0);
	}
 getInfo(candle) {
		let cond = candle.open > candle.close;
		return {
			color: cond ? this.bearColor : this.bullColor,
			type: cond ? 'bear' : 'bull'
		}
	}
	candleBody(type = 'bull', color = this.bullColor) {
		if (type == 'bull') {
			this.bullColor = color
		} else {
			this.bearColor = color
		}
		this.drawCandles();
	}
	candleLineEdit(type, color) {
		if (type == 'bull') {
			if (this.bullCandleLineColor == 'color') return;
			this.bullCandleLineColor = color;
		} else {
			if (this.bearCandleLineColor == 'color') return;
			this.bearCandleLineColor = color;
		}
		this.drawCandles();
	}
}
const candles = new CandleSticks();

const canvas2 = iframe.getElementById('canvas2');
const ctx2 = canvas2.getContext('2d');

function fillPriceCanvas() {
	let labelSpacing = Math.floor(height/50); // 50 is the bottom spacing for each price before the next one
	let h = Math.ceil(high);
	let difference = (h - Math.floor(low)) / labelSpacing;
	let price = h;
	
	ctx2.clearRect(0, 0, canvas2.width, canvas2.height);
	ctx2.beginPath();
	ctx2.fillStyle = 'white';

	for (let i = 0; i < labelSpacing+1; i++) {
		ctx2.fillText('-' + price.toFixed(2), 0, (h - price.toFixed(2)) * px);
		price -= difference;
	}
	
	ctx2.font = '10px sans-serif';
	ctx2.textBaseline = 'middle';
	ctx2.fillRect(0, ((h - priceArray[len].close) * px) - 7.5, 60, 15);
	ctx2.fill();
	ctx2.fillStyle = 'black';
	ctx2.fillText('-' + priceArray[len].close, 0, (h - priceArray[len].close) * px);
}

chartDiv.scrollLeft = mainElem.scrollWidth;

let chartType = document.getElementById('chartType');
for (let child of chartType.children) {
	child.classList.add('hide');
	if (child.dataset.type !== type) {
		child.classList.remove('hide');
	}
}

function getParent(elem, tagName, exception) {
	if (elem.nodeName.toLowerCase() == tagName) return elem;
	if (exception && elem.nodeName.toLowerCase() == exception) return;
	let child;
	if (elem.parentElement) child = getParent(elem.parentElement, tagName, exception);
	return child;
}

chartType.addEventListener('click', function(e) {
	let svg = getParent(e.target, 'svg', 'button');
	type = svg.dataset.type;
	for (let icon of svg.parentElement.children) icon.classList.remove('hide');
	svg.classList.add('hide');
	localStorage.setItem('tpe', type);
	candles.drawCandles();
});

async function fetchData(url) {
	url = 'https://www.alphavantage.co/query?function=' + url;
	//if(url) return;
	/*	fetch(url)
		.then(res => res.json())
		.then(res => {*/
	let result = {
		"Meta Data": {
			"1. Information": "Weekly Prices (open, high, low, close) and Volumes",
			"2. Symbol": "IBM",
			"3. Last Refreshed": "2023-06-16",
			"4. Time Zone": "US/Eastern"
		},
		"Weekly Time Series": {
			"2023-06-16": {
				"1. open": "136.0000",
				"2. high": "139.4690",
				"3. low": "135.8216",
				"4. close": "137.4800",
				"5. volume": "24159260"
			},
			"2023-06-09": {
				"1. open": "133.1200",
				"2. high": "136.1000",
				"3. low": "131.8800",
				"4. close": "135.3000",
				"5. volume": "21174178"
			},
			"2023-06-02": {
				"1. open": "129.5600",
				"2. high": "133.1200",
				"3. low": "127.4600",
				"4. close": "132.4200",
				"5. volume": "24339245"
			},
			"2023-05-26": {
				"1. open": "127.5000",
				"2. high": "129.6600",
				"3. low": "125.0100",
				"4. close": "128.8900",
				"5. volume": "21029979"
			},
			"2023-05-19": {
				"1. open": "123.0000",
				"2. high": "128.2900",
				"3. low": "122.3400",
				"4. close": "127.2600",
				"5. volume": "18284524"
			},
			"2023-05-12": {
				"1. open": "123.7600",
				"2. high": "123.9200",
				"3. low": "120.5500",
				"4. close": "122.8400",
				"5. volume": "20404364"
			},
			"2023-05-05": {
				"1. open": "126.3500",
				"2. high": "126.7500",
				"3. low": "121.7563",
				"4. close": "123.6500",
				"5. volume": "21164660"
			},
			"2023-04-28": {
				"1. open": "125.5500",
				"2. high": "127.2500",
				"3. low": "124.5600",
				"4. close": "126.4100",
				"5. volume": "20644224"
			},
			"2023-04-21": {
				"1. open": "128.3000",
				"2. high": "130.9800",
				"3. low": "125.2700",
				"4. close": "125.7300",
				"5. volume": "30341128"
			},
			"2023-04-14": {
				"1. open": "129.8300",
				"2. high": "131.1050",
				"3. low": "126.0000",
				"4. close": "128.1400",
				"5. volume": "19506500"
			},
			"2023-04-06": {
				"1. open": "130.9700",
				"2. high": "132.6100",
				"3. low": "130.3150",
				"4. close": "130.5000",
				"5. volume": "13172262"
			},
			"2023-03-31": {
				"1. open": "126.4700",
				"2. high": "131.4800",
				"3. low": "126.4700",
				"4. close": "131.0900",
				"5. volume": "20779522"
			},
			"2023-03-24": {
				"1. open": "124.3100",
				"2. high": "127.2150",
				"3. low": "122.6000",
				"4. close": "125.2900",
				"5. volume": "20458253"
			},
			"2023-03-17": {
				"1. open": "125.1500",
				"2. high": "128.1900",
				"3. low": "121.7100",
				"4. close": "123.6900",
				"5. volume": "66132690"
			},
			"2023-03-10": {
				"1. open": "129.6400",
				"2. high": "130.8600",
				"3. low": "125.1300",
				"4. close": "125.4500",
				"5. volume": "20761401"
			},
			"2023-03-03": {
				"1. open": "131.4200",
				"2. high": "131.8700",
				"3. low": "127.7100",
				"4. close": "129.6400",
				"5. volume": "17865677"
			},
			"2023-02-24": {
				"1. open": "134.0000",
				"2. high": "134.3850",
				"3. low": "128.8600",
				"4. close": "130.5700",
				"5. volume": "14198950"
			},
			"2023-02-17": {
				"1. open": "136.0000",
				"2. high": "137.3900",
				"3. low": "133.8900",
				"4. close": "135.0200",
				"5. volume": "16543870"
			},
			"2023-02-10": {
				"1. open": "135.8300",
				"2. high": "136.7400",
				"3. low": "133.3400",
				"4. close": "135.6000",
				"5. volume": "22140989"
			},
			"2023-02-03": {
				"1. open": "134.3200",
				"2. high": "136.9500",
				"3. low": "132.8000",
				"4. close": "136.9400",
				"5. volume": "27874571"
			},
			"2023-01-27": {
				"1. open": "141.4000",
				"2. high": "142.9850",
				"3. low": "132.9800",
				"4. close": "134.3900",
				"5. volume": "43345140"
			},
			"2023-01-20": {
				"1. open": "146.4200",
				"2. high": "147.1800",
				"3. low": "139.7500",
				"4. close": "141.2000",
				"5. volume": "21419368"
			},
			"2023-01-13": {
				"1. open": "144.0800",
				"2. high": "146.6600",
				"3. low": "142.9000",
				"4. close": "145.8900",
				"5. volume": "14580596"
			},
			"2023-01-06": {
				"1. open": "141.1000",
				"2. high": "144.2500",
				"3. low": "140.0100",
				"4. close": "143.7000",
				"5. volume": "13648755"
			},
			"2022-12-30": {
				"1. open": "141.7300",
				"2. high": "142.8100",
				"3. low": "139.4500",
				"4. close": "140.8900",
				"5. volume": "10477419"
			},
			"2022-12-23": {
				"1. open": "140.1600",
				"2. high": "143.0900",
				"3. low": "137.1950",
				"4. close": "141.6500",
				"5. volume": "19663576"
			},
			"2022-12-16": {
				"1. open": "147.8200",
				"2. high": "153.2100",
				"3. low": "138.9700",
				"4. close": "140.1600",
				"5. volume": "33572382"
			},
			"2022-12-09": {
				"1. open": "147.9400",
				"2. high": "149.1530",
				"3. low": "146.2900",
				"4. close": "147.0500",
				"5. volume": "15316930"
			},
			"2022-12-02": {
				"1. open": "147.9800",
				"2. high": "150.0100",
				"3. low": "145.6700",
				"4. close": "148.6700",
				"5. volume": "20066941"
			},
			"2022-11-25": {
				"1. open": "147.5500",
				"2. high": "150.4600",
				"3. low": "146.4500",
				"4. close": "148.3700",
				"5. volume": "16271883"
			},
			"2022-11-18": {
				"1. open": "142.6300",
				"2. high": "148.3100",
				"3. low": "142.0000",
				"4. close": "147.6400",
				"5. volume": "22034930"
			},
			"2022-11-11": {
				"1. open": "136.6400",
				"2. high": "144.1300",
				"3. low": "136.5100",
				"4. close": "143.1700",
				"5. volume": "25066114"
			},
			"2022-11-04": {
				"1. open": "138.0600",
				"2. high": "140.1700",
				"3. low": "133.9700",
				"4. close": "136.9600",
				"5. volume": "22491556"
			},
			"2022-10-28": {
				"1. open": "130.9000",
				"2. high": "138.8615",
				"3. low": "129.8500",
				"4. close": "138.5100",
				"5. volume": "26667185"
			},
			"2022-10-21": {
				"1. open": "121.8000",
				"2. high": "130.8450",
				"3. low": "121.4300",
				"4. close": "129.9000",
				"5. volume": "37309868"
			},
			"2022-10-14": {
				"1. open": "119.7900",
				"2. high": "122.5400",
				"3. low": "115.5450",
				"4. close": "120.0400",
				"5. volume": "22973512"
			},
			"2022-10-07": {
				"1. open": "120.1600",
				"2. high": "126.4600",
				"3. low": "118.0700",
				"4. close": "118.8200",
				"5. volume": "21614952"
			},
			"2022-09-30": {
				"1. open": "122.3000",
				"2. high": "124.2600",
				"3. low": "118.6100",
				"4. close": "118.8100",
				"5. volume": "22265241"
			},
			"2022-09-23": {
				"1. open": "126.4900",
				"2. high": "128.0600",
				"3. low": "121.7400",
				"4. close": "122.7100",
				"5. volume": "19442664"
			},
			"2022-09-16": {
				"1. open": "130.3300",
				"2. high": "130.9900",
				"3. low": "123.8300",
				"4. close": "127.2700",
				"5. volume": "27107187"
			},
			"2022-09-09": {
				"1. open": "127.8000",
				"2. high": "129.4900",
				"3. low": "126.2800",
				"4. close": "129.1900",
				"5. volume": "12004834"
			},
			"2022-09-02": {
				"1. open": "129.9900",
				"2. high": "131.4200",
				"3. low": "127.2400",
				"4. close": "127.7900",
				"5. volume": "15086608"
			},
			"2022-08-26": {
				"1. open": "137.6500",
				"2. high": "137.8500",
				"3. low": "130.3400",
				"4. close": "130.3800",
				"5. volume": "16982107"
			},
			"2022-08-19": {
				"1. open": "132.9600",
				"2. high": "139.3400",
				"3. low": "132.2400",
				"4. close": "138.3700",
				"5. volume": "17105977"
			},
			"2022-08-12": {
				"1. open": "133.1000",
				"2. high": "134.0900",
				"3. low": "129.1200",
				"4. close": "134.0100",
				"5. volume": "17254110"
			},
			"2022-08-05": {
				"1. open": "130.7500",
				"2. high": "132.8620",
				"3. low": "130.5100",
				"4. close": "132.4800",
				"5. volume": "17400572"
			},
			"2022-07-29": {
				"1. open": "128.4400",
				"2. high": "131.0000",
				"3. low": "127.5800",
				"4. close": "130.7900",
				"5. volume": "22223785"
			},
			"2022-07-22": {
				"1. open": "140.1500",
				"2. high": "140.3100",
				"3. low": "125.1300",
				"4. close": "128.2500",
				"5. volume": "66246811"
			},
			"2022-07-15": {
				"1. open": "140.6200",
				"2. high": "141.8700",
				"3. low": "135.0200",
				"4. close": "139.9200",
				"5. volume": "21089228"
			},
			"2022-07-08": {
				"1. open": "139.9700",
				"2. high": "141.3250",
				"3. low": "135.2700",
				"4. close": "140.4700",
				"5. volume": "16229131"
			},
			"2022-07-01": {
				"1. open": "142.2600",
				"2. high": "144.1550",
				"3. low": "139.2600",
				"4. close": "141.1200",
				"5. volume": "21052787"
			},
			"2022-06-24": {
				"1. open": "135.9000",
				"2. high": "142.3700",
				"3. low": "135.9000",
				"4. close": "142.0600",
				"5. volume": "23921844"
			},
			"2022-06-17": {
				"1. open": "133.9700",
				"2. high": "138.4500",
				"3. low": "132.8500",
				"4. close": "135.0200",
				"5. volume": "34055621"
			},
			"2022-06-10": {
				"1. open": "142.9800",
				"2. high": "144.7300",
				"3. low": "135.2500",
				"4. close": "136.1900",
				"5. volume": "18914084"
			},
			"2022-06-03": {
				"1. open": "138.2000",
				"2. high": "142.5794",
				"3. low": "136.8100",
				"4. close": "141.1800",
				"5. volume": "19861566"
			},
			"2022-05-27": {
				"1. open": "129.5000",
				"2. high": "139.7394",
				"3. low": "129.4200",
				"4. close": "139.2700",
				"5. volume": "19618090"
			},
			"2022-05-20": {
				"1. open": "133.1000",
				"2. high": "138.3700",
				"3. low": "125.8000",
				"4. close": "128.4800",
				"5. volume": "27360441"
			},
			"2022-05-13": {
				"1. open": "134.4100",
				"2. high": "136.3450",
				"3. low": "128.4300",
				"4. close": "133.6000",
				"5. volume": "31171489"
			},
			"2022-05-06": {
				"1. open": "133.0000",
				"2. high": "137.9900",
				"3. low": "130.8900",
				"4. close": "137.6700",
				"5. volume": "27079586"
			},
			"2022-04-29": {
				"1. open": "137.5900",
				"2. high": "139.8700",
				"3. low": "132.0000",
				"4. close": "132.2100",
				"5. volume": "24560464"
			},
			"2022-04-22": {
				"1. open": "126.6000",
				"2. high": "141.8800",
				"3. low": "125.5300",
				"4. close": "138.2500",
				"5. volume": "47056765"
			},
			"2022-04-14": {
				"1. open": "127.9500",
				"2. high": "130.5800",
				"3. low": "124.9100",
				"4. close": "126.5600",
				"5. volume": "15342641"
			},
			"2022-04-08": {
				"1. open": "130.2600",
				"2. high": "131.2300",
				"3. low": "126.7300",
				"4. close": "127.7300",
				"5. volume": "16553021"
			},
			"2022-04-01": {
				"1. open": "130.8200",
				"2. high": "133.0800",
				"3. low": "128.0600",
				"4. close": "130.1500",
				"5. volume": "19183786"
			},
			"2022-03-25": {
				"1. open": "129.0000",
				"2. high": "131.4000",
				"3. low": "127.4000",
				"4. close": "131.3500",
				"5. volume": "15440952"
			},
			"2022-03-18": {
				"1. open": "124.4500",
				"2. high": "128.9300",
				"3. low": "122.6850",
				"4. close": "128.7600",
				"5. volume": "22675632"
			},
			"2022-03-11": {
				"1. open": "126.4700",
				"2. high": "128.3450",
				"3. low": "123.1250",
				"4. close": "123.9600",
				"5. volume": "23784887"
			},
			"2022-03-04": {
				"1. open": "122.2100",
				"2. high": "127.3500",
				"3. low": "120.7000",
				"4. close": "126.6200",
				"5. volume": "26131658"
			},
			"2022-02-25": {
				"1. open": "124.2000",
				"2. high": "125.0000",
				"3. low": "118.8100",
				"4. close": "124.1800",
				"5. volume": "20417161"
			},
			"2022-02-18": {
				"1. open": "132.5900",
				"2. high": "132.6500",
				"3. low": "123.6100",
				"4. close": "124.3500",
				"5. volume": "26022824"
			},
			"2022-02-11": {
				"1. open": "137.4500",
				"2. high": "138.3500",
				"3. low": "132.3800",
				"4. close": "132.6900",
				"5. volume": "23489144"
			},
			"2022-02-04": {
				"1. open": "134.0900",
				"2. high": "138.8200",
				"3. low": "132.3000",
				"4. close": "137.1500",
				"5. volume": "27665550"
			},
			"2022-01-28": {
				"1. open": "127.9900",
				"2. high": "137.3361",
				"3. low": "124.1930",
				"4. close": "134.5000",
				"5. volume": "52800401"
			},
			"2022-01-21": {
				"1. open": "132.9500",
				"2. high": "133.9000",
				"3. low": "129.2700",
				"4. close": "129.3500",
				"5. volume": "20520487"
			},
			"2022-01-14": {
				"1. open": "134.4700",
				"2. high": "136.2000",
				"3. low": "127.9700",
				"4. close": "134.2100",
				"5. volume": "32044685"
			},
			"2022-01-07": {
				"1. open": "134.0700",
				"2. high": "142.2000",
				"3. low": "132.5100",
				"4. close": "134.8300",
				"5. volume": "36013766"
			},
			"2021-12-31": {
				"1. open": "130.6300",
				"2. high": "134.9900",
				"3. low": "129.9500",
				"4. close": "133.6600",
				"5. volume": "18454937"
			},
			"2021-12-23": {
				"1. open": "125.7200",
				"2. high": "130.9600",
				"3. low": "124.7000",
				"4. close": "130.6300",
				"5. volume": "17369625"
			},
			"2021-12-17": {
				"1. open": "123.7600",
				"2. high": "128.6400",
				"3. low": "120.7900",
				"4. close": "127.4000",
				"5. volume": "35216850"
			},
			"2021-12-10": {
				"1. open": "119.4000",
				"2. high": "125.3300",
				"3. low": "119.4000",
				"4. close": "124.0900",
				"5. volume": "25031512"
			},
			"2021-12-03": {
				"1. open": "118.6200",
				"2. high": "119.6100",
				"3. low": "116.4500",
				"4. close": "118.8400",
				"5. volume": "36059651"
			},
			"2021-11-26": {
				"1. open": "116.0000",
				"2. high": "118.8100",
				"3. low": "114.5600",
				"4. close": "115.8100",
				"5. volume": "17875027"
			},
			"2021-11-19": {
				"1. open": "119.5400",
				"2. high": "120.1600",
				"3. low": "115.2700",
				"4. close": "116.0500",
				"5. volume": "24272797"
			},
			"2021-11-12": {
				"1. open": "123.9850",
				"2. high": "124.7800",
				"3. low": "118.7800",
				"4. close": "118.9600",
				"5. volume": "29109912"
			},
			"2021-11-05": {
				"1. open": "125.0500",
				"2. high": "127.2900",
				"3. low": "119.9000",
				"4. close": "123.6100",
				"5. volume": "29791780"
			},
			"2021-10-29": {
				"1. open": "127.5300",
				"2. high": "128.6500",
				"3. low": "124.6200",
				"4. close": "125.1000",
				"5. volume": "34288134"
			},
			"2021-10-22": {
				"1. open": "144.0000",
				"2. high": "144.9400",
				"3. low": "126.6110",
				"4. close": "127.8800",
				"5. volume": "59731582"
			},
			"2021-10-15": {
				"1. open": "143.5000",
				"2. high": "144.8500",
				"3. low": "139.6600",
				"4. close": "144.6100",
				"5. volume": "16262687"
			},
			"2021-10-08": {
				"1. open": "142.7400",
				"2. high": "146.0000",
				"3. low": "140.8900",
				"4. close": "143.2200",
				"5. volume": "27211291"
			},
			"2021-10-01": {
				"1. open": "137.9600",
				"2. high": "143.9700",
				"3. low": "136.4400",
				"4. close": "143.3200",
				"5. volume": "23824191"
			},
			"2021-09-24": {
				"1. open": "133.9000",
				"2. high": "138.4800",
				"3. low": "132.7800",
				"4. close": "137.4900",
				"5. volume": "18425230"
			},
			"2021-09-17": {
				"1. open": "138.4000",
				"2. high": "138.9900",
				"3. low": "135.0500",
				"4. close": "135.2300",
				"5. volume": "20130213"
			},
			"2021-09-10": {
				"1. open": "139.6500",
				"2. high": "139.7900",
				"3. low": "137.0000",
				"4. close": "137.0200",
				"5. volume": "13754250"
			},
			"2021-09-03": {
				"1. open": "139.5000",
				"2. high": "140.9400",
				"3. low": "138.8150",
				"4. close": "139.5800",
				"5. volume": "13345045"
			},
			"2021-08-27": {
				"1. open": "139.6200",
				"2. high": "140.8000",
				"3. low": "138.4000",
				"4. close": "139.4100",
				"5. volume": "12376600"
			},
			"2021-08-20": {
				"1. open": "143.2300",
				"2. high": "143.7400",
				"3. low": "137.2100",
				"4. close": "139.1100",
				"5. volume": "16189007"
			},
			"2021-08-13": {
				"1. open": "142.2000",
				"2. high": "143.5800",
				"3. low": "140.3400",
				"4. close": "143.1800",
				"5. volume": "18462255"
			},
			"2021-08-06": {
				"1. open": "141.4500",
				"2. high": "144.7000",
				"3. low": "141.0300",
				"4. close": "144.0900",
				"5. volume": "16428567"
			},
			"2021-07-30": {
				"1. open": "141.3900",
				"2. high": "143.6400",
				"3. low": "140.7900",
				"4. close": "140.9600",
				"5. volume": "16120616"
			}
		}
	};
	fetching = false;
	spinner.classList.add('hide');
/*
	priceArray = [];

	let series = result["Weekly Time Series"];
	let index = 0;
	for (let key in series) {
		index++;
		if (index > 50) break;
		for (let val in series[key]) {
			series[key][val.slice(3)] = series[key][val];
			delete series[key][val];
		}
		series[key].time = key;
		priceArray.push(series[key]);
	}
	*/
		priceArray = [
			{
				open: 1826.04,
				high: 1826.17,
				low: 1825.75,
				close: 1825.86,
				time: '09:30',
				fullTime: '2022.02.08 09:30'
		},
			{
				open: 1825.88,
				high: 1825.93,
				low: 1825.07,
				close: 1825.23,
				time: '10:00',
				fullTime: '2022.02.08 10:00'
		},
			{
				open: 1825.18,
				high: 1825.49,
				low: 1824.98,
				close: 1825.37,
				time: '10:30',
				fullTime: '2022.02.08 10:30'
		},
			{
				open: 1825.34,
				high: 1826.69,
				low: 1825.07,
				close: 1825.46,
				time: '11:00',
				fullTime: '2022.02.08 11:00'
		},
			{
				open: 1825.46,
				high: 1826.75,
				low: 1825.46,
				close: 1826.67,
				time: '11:30',
				fullTime: '2022.02.08 11:30'
		},
			{
				open: 1826.68,
				high: 1827.76,
				low: 1826.45,
				close: 1827.52,
				time: '12:00',
				fullTime: '2022.02.08 12:00'
		},
			{
				open: 1827.56,
				high: 1828.09,
				low: 1826.35,
				close: 1827.46,
				time: '12:30',
				fullTime: '2022.02.08 12:30'
		},
			{
				open: 1827.47,
				high: 1828.65,
				low: 1827.47,
				close: 1828.30,
				time: '13:00',
				fullTime: '2022.02.08 13:00'
		},
			{
				open: 1828.30,
				high: 1828.62,
				low: 1827.55,
				close: 1828.01,
				time: '13:30',
				fullTime: '2022.02.08 13:30'
		},
			{
				open: 1828.01,
				high: 1828.39,
				low: 1827.18,
				close: 1827.29,
				time: '14:00',
				fullTime: '2022.02.08 14:00'
		},
			{
				open: 1827.30,
				high: 1827.82,
				low: 1827.15,
				close: 1827.19,
				time: '14:30',
				fullTime: '2022.02.08 14:30'
		},
			{
				open: 1827.18,
				high: 1828.01,
				low: 1827.06,
				close: 1827.83,
				time: '15:00',
				fullTime: '2022.02.08 15:00'
		},
			{
				open: 1827.83,
				high: 1829.06,
				low: 1827.36,
				close: 1827.36,
				time: '15:30',
				fullTime: '2022.02.08 15:30'
		},
			{
				open: 1827.36,
				high: 1828.11,
				low: 1826.50,
				close: 1826.54,
				time: '16:00',
				fullTime: '2022.02.08 16:00'
		},
			{
				open: 1826.52,
				high: 1827.67,
				low: 1826.42,
				close: 1827.44,
				time: '16:30',
				fullTime: '2022.02.08 16:30'
		},
			{
				open: 1827.36,
				high: 1828.20,
				low: 1826.69,
				close: 1826.81,
				time: '17:00',
				fullTime: '2022.02.08 17:00'
		},
			{
				open: 1826.81,
				high: 1826.85,
				low: 1825.67,
				close: 1826.67,
				time: '17:30',
				fullTime: '2022.02.08 17:30'
		},
			{
				open: 1826.64,
				high: 1828.15,
				low: 1826.27,
				close: 1827.18,
				time: '18:00',
				fullTime: '2022.02.08 18:00'
		},
			{
				open: 1827.13,
				high: 1827.72,
				low: 1826.15,
				close: 1826.35,
				time: '18:30',
				fullTime: '2022.02.08 18:30'
		},
			{
				open: 1826.30,
				high: 1827.63,
				low: 1826.13,
				close: 1827.41,
				time: '19:00',
				fullTime: '2022.02.08 19:00'
		},
			{
				open: 1827.39,
				high: 1827.78,
				low: 1826.83,
				close: 1827.68,
				time: '19:30',
				fullTime: '2022.02.08 19:30'
		},
			{
				open: 1827.66,
				high: 1828.26,
				low: 1826.88,
				close: 1827.53,
				time: '20:00',
				fullTime: '2022.02.08 20:00'
		},
			{
				open: 1827.53,
				high: 1828.01,
				low: 1826.69,
				close: 1826.79,
				time: '20:30',
				fullTime: '2022.02.08 20:30'
		},
			{
				open: 1826.75,
				high: 1827.05,
				low: 1825.98,
				close: 1826.52,
				time: '21:00',
				fullTime: '2022.02.08 21:00'
		},
			{
				open: 1826.52,
				high: 1827.19,
				low: 1825.59,
				close: 1826.81,
				time: '21:30',
				fullTime: '2022.02.08 21:30'
		},
			{
				open: 1826.80,
				high: 1827.69,
				low: 1826.80,
				close: 1826.90,
				time: '22:00',
				fullTime: '2022.02.08 22:00'
		},
			{
				open: 1826.91,
				high: 1828.33,
				low: 1826.12,
				close: 1828.09,
				time: '22:30',
				fullTime: '2022.02.08 22:30'
		},
			{
				open: 1828.11,
				high: 1832.48,
				low: 1828.11,
				close: 1830.66,
				time: '23:00',
				fullTime: '2022.02.08 23:00'
		},
			{
				open: 1830.72,
				high: 1831.30,
				low: 1825.84,
				close: 1825.89,
				time: '23:30',
				fullTime: '2022.02.08 23:30'
		},
			{
				open: 1825.95,
				high: 1828.42,
				low: 1824.58,
				close: 1827.71,
				time: '00:00',
				fullTime: '2022.02.08 00:00'
		},
			{
				open: 1826.04,
				high: 1826.17,
				low: 1825.75,
				close: 1825.86,
				time: '00:30',
				fullTime: '2022.02.08 00:30'
		},
			{
				open: 1825.88,
				high: 1825.93,
				low: 1825.07,
				close: 1825.23,
				time: '01:00',
				fullTime: '2022.02.08 01:00'
		},
			{
				open: 1825.18,
				high: 1825.49,
				low: 1824.98,
				close: 1825.37,
				time: '01:30',
				fullTime: '2022.02.08 01:30'
		},
			{
				open: 1825.34,
				high: 1826.69,
				low: 1825.07,
				close: 1825.46,
				time: '02:00',
				fullTime: '2022.02.08 02:00'
		},
			{
				open: 1825.46,
				high: 1826.75,
				low: 1825.46,
				close: 1826.67,
				time: '02:30',
				fullTime: '2022.02.08 02:30'
		},
			{
				open: 1826.68,
				high: 1827.76,
				low: 1826.45,
				close: 1827.52,
				time: '03:00',
				fullTime: '2022.02.08 03:00'
		},
			{
				open: 1827.56,
				high: 1828.09,
				low: 1826.35,
				close: 1827.46,
				time: '03:30',
				fullTime: '2022.02.08 03:30'
		},
			{
				open: 1827.47,
				high: 1828.65,
				low: 1827.47,
				close: 1828.30,
				time: '04:00',
				fullTime: '2022.02.08 04:00'
		},
			{
				open: 1828.30,
				high: 1828.62,
				low: 1827.55,
				close: 1828.01,
				time: '04:30',
				fullTime: '2022.02.08 04:30'
		},
			{
				open: 1828.01,
				high: 1828.39,
				low: 1827.18,
				close: 1827.29,
				time: '05:00',
				fullTime: '2022.02.08 05:00'
		},
			{
				open: 1827.30,
				high: 1827.82,
				low: 1827.15,
				close: 1827.19,
				time: '05:30',
				fullTime: '2022.02.08 05:30'
		},
			{
				open: 1827.18,
				high: 1828.01,
				low: 1827.06,
				close: 1827.83,
				time: '06:00',
				fullTime: '2022.02.08 06:00'
		},
			{
				open: 1827.83,
				high: 1829.06,
				low: 1827.36,
				close: 1827.36,
				time: '06:30',
				fullTime: '2022.02.08 06:30'
		},
			{
				open: 1827.36,
				high: 1828.11,
				low: 1826.50,
				close: 1826.54,
				time: '07:00',
				fullTime: '2022.02.08 07:00'
		},
			{
				open: 1826.52,
				high: 1827.67,
				low: 1826.42,
				close: 1827.44,
				time: '07:30',
				fullTime: '2022.02.08 07:30'
		},
			{
				open: 1827.36,
				high: 1828.20,
				low: 1826.69,
				close: 1826.81,
				time: '08:00',
				fullTime: '2022.02.08 08:00'
		},
			{
				open: 1826.81,
				high: 1826.85,
				low: 1825.67,
				close: 1826.67,
				time: '08:30',
				fullTime: '2022.02.08 08:30'
		},
			{
				open: 1826.64,
				high: 1828.15,
				low: 1826.27,
				close: 1827.18,
				time: '09:00',
				fullTime: '2022.02.08 09:00'
		},
			{
				open: 1827.13,
				high: 1827.72,
				low: 1826.15,
				close: 1826.35,
				time: '09:30',
				fullTime: '2022.02.08 09:30'
		},
			{
				open: 1826.30,
				high: 1827.63,
				low: 1826.13,
				close: 1827.41,
				time: '10:00',
				fullTime: '2022.02.08 10:00'
		},
			{
				open: 1827.39,
				high: 1827.78,
				low: 1826.83,
				close: 1827.68,
				time: '10:30',
				fullTime: '2022.02.08 10:30'
		},
			{
				open: 1827.66,
				high: 1828.26,
				low: 1826.88,
				close: 1827.53,
				time: '11:00',
				fullTime: '2022.02.08 11:00'
		},
			{
				open: 1827.53,
				high: 1828.01,
				low: 1826.69,
				close: 1826.79,
				time: '11:30',
				fullTime: '2022.02.08 11:30'
		},
			{
				open: 1826.75,
				high: 1827.05,
				low: 1825.98,
				close: 1826.52,
				time: '12:00',
				fullTime: '2022.02.08 12:00'
		},
			{
				open: 1826.52,
				high: 1827.19,
				low: 1825.59,
				close: 1826.81,
				time: '12:30',
				fullTime: '2022.02.08 12:30'
		},
			{
				open: 1826.80,
				high: 1827.69,
				low: 1826.80,
				close: 1826.90,
				time: '13:00',
				fullTime: '2022.02.08 13:00'
		},
			{
				open: 1826.91,
				high: 1828.33,
				low: 1826.12,
				close: 1828.09,
				time: '13:30',
				fullTime: '2022.02.08 13:30'
		},
			{
				open: 1828.11,
				high: 1833,
				low: 1828.11,
				close: 1830.66,
				time: '14:00',
				fullTime: '2022.02.08 14:00'
		},
			{
				open: 1830.72,
				high: 1831.30,
				low: 1825.84,
				close: 1825.89,
				time: '14:30',
				fullTime: '2022.02.08 14:30'
		},
			{
				open: 1825.95,
				high: 1828.42,
				low: 1824,
				close: 1827.71,
				time: '15:00',
				fullTime: '2022.02.08 15:00'
		},
			{
				open: 1827.89,
				high: 1831.03,
				low: 1826.61,
				close: 1828.65,
				time: '15:30',
				fullTime: '2022.02.08 15:30'
		},
			{
				open: 1828.62,
				high: 1831.52,
				low: 1827.31,
				close: 1831.18,
				time: '16:00',
				fullTime: '2022.02.08 16:00'
		},
			{
				open: 1831.17,
				high: 1831.33,
				low: 1827.95,
				close: 1828.47,
				time: '16:30',
				fullTime: '2022.02.08 16:30'
		},
			{
				open: 1828.46,
				high: 1830.77,
				low: 1828.33,
				close: 1830.61,
				time: '17:00',
				fullTime: '2022.02.08 17:00'
		},
			{
				open: 1830.62,
				high: 1833.36,
				low: 1830.58,
				close: 1832.51,
				time: '17:30',
				fullTime: '2022.02.08 17:30'
		},
			{
				open: 1832.53,
				high: 1836,
				low: 1832.49,
				close: 1835.44,
				time: '18:00',
				fullTime: '2022.02.08 18:00'
		}
	]; // = res;
	
	//priceArray.reverse();
	
	len = priceArray.length - 1;

	domOpen.textContent = priceArray[len].open;
	domHigh.textContent = priceArray[len].high;
	domLow.textContent = priceArray[len].low;
	domClose.textContent = priceArray[len].close;

	// Get highest and lowest point in price
	high = Math.max.apply(null, priceArray.map(a => a.high));
	low = Math.min.apply(null, priceArray.map(a => a.low));
	high += 2;
	low -= 2;
	candles.drawCandles();
	fillPriceCanvas();
	/*	})
		.catch(error => console.log(error.stack));*/
}
fetchData('');

async function fetchQuotes(url) {
	/*	fetch(url)
			.then(res => res.json())
			.then(res => {*/
	let quotes_list = document.getElementsByClassName('quotes-list')[0];

	let objArr = [
		{
			pair: 'XAUUSD',
			currentTime: '11:17:00',
			bid: '151.25',
			ask: '151.36',
			low: '150.89',
			high: '151.48'
			},
		{
			pair: 'XAUUSD',
			currentTime: '11:17:00',
			bid: '151.25',
			ask: '151.36',
			low: '150.89',
			high: '151.48'
			}
			];

	function createList(list) {
		let li = cEl('li', { className: "flex-cent border-bottom border-light p-2" },
			cEl('div', { className: "w-1/3 px-2" },
				cEl('div', {},
					cEl('b', { textContent: list.pair })
				),
				cEl('small', { textContent: list.currentTime })
			),
			cEl('div', {},
				cEl('div', { className: 'flex center w-100 color-bear' },
					cEl('div', { className: 'px-2 col-2' },
						cEl('b', { textContent: list.bid })
					),
					cEl('div', { className: 'px-2 col-2' },
						cEl('b', { textContent: list.ask })
					)
				),
				cEl('div', { className: 'flex center p-1' },
					cEl('small', { className: 'px-2 col-2', textContent: 'Low: ' + list.low }),
					cEl('small', { className: 'px-2 col-2', textContent: 'High: ' + list.high })
				)
			)
		);
		return li;
	}

	objArr.forEach(obj => quotes_list.append(createList(obj)));
	/*	})
		.catch(error => console.log(error.stack));*/
}
fetchQuotes();

/* Enable mouse wheel horizontal scrolling */
chartDiv.addEventListener('wheel', function(e) {
	if (e.deltaY >= 100) chartDiv.scrollLeft -= 50;
	if (e.deltaY <= -100) chartDiv.scrollLeft += 50;
});

/* Display tooltip with candle information */
const title = document.getElementsByClassName('title')[0];

let addedListener = false;
const isMobile = window.outerWidth < 768;

canvas.addEventListener(isMobile ? 'click' : 'pointermove', function(e) {
	ctx.clearRect(0, 0, canvas.width, canvas.height);
	ctx.drawImage(candles.chartImg, 0, 0);
	
	const x = e.clientX + chartDiv.scrollLeft;
	const candle = candlesX.findIndex(num => num >= x);
	const info = priceArray[candle == -1 ? len : candle+1];
	
	if (candle) {
		domOpen.textContent = info.open;
		domClose.textContent = info.close;
		domHigh.textContent = info.high;
		domLow.textContent = info.low;

		ctx.beginPath();
		ctx.strokeStyle = 'white';
		ctx.lineWidth = 1;
		drawLine(candlesX[candle], 0, candlesX[candle], canvas.height -25);
		drawLine(0, e.clientY, canvas.width, e.clientY);
		ctx.beginPath();
		ctx.strokeStyle = 'black';
		ctx.fillStyle = 'black';
		ctx.arc(candlesX[candle], e.clientY, 5, 0, 2 * Math.PI);
		ctx.stroke();
		ctx.fill();
	}

	title.style.top = e.clientY + "px";
	title.style.left = e.clientX+candleWidth/2 + "px";
	title.innerText = `Open: ${info.open}\nHigh: ${info.high}\nLow: ${info.low}\nClose: ${info.close}\nTime: ${info.fullTime}`;

	title.classList.add('block');
	if (!addedListener) {
		(isMobile ? document.body : canvas).addEventListener(isMobile ? 'click' : 'pointerout', function callback(e) {
			if (isMobile && e.target != canvas) {
				title.classList.remove('block');
				(isMobile ? document.body : canvas).removeEventListener(isMobile ? 'click' : 'pointerout', callback);
				ctx.clearRect(0, 0, canvas.width, canvas.height);
				ctx.drawImage(candles.chartImg, 0, 0);

				let currentInfo = priceArray[len];
				domOpen.textContent = currentInfo.open;
				domClose.textContent = currentInfo.close;
				domHigh.textContent = currentInfo.high;
				domLow.textContent = currentInfo.low;
			}
			addedListener = false;
		});
		addedListener = true;
	}
});

let aside = document.getElementsByTagName('aside')[0];

let appLoc;
document.getElementById('quotes').addEventListener('click', function() {
	setTimeout(() => aside.classList.remove('hide-aside'), 1);
	aside.classList.remove('hide');
	history.pushState({ page: 1 }, 'Quotes list', "/quotes");
	appLoc = location.href;
});

document.getElementsByClassName('arrow')[0].addEventListener('click', function() {
	history.back();
});

window.addEventListener("popstate", (event) => {
	if (new URL(appLoc).pathname == '/quotes') {
		setTimeout(() => aside.classList.add('hide'), 800);
		aside.classList.add('hide-aside');
	}
});