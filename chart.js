iframe = iframe.contentDocument;

const domHigh = document.getElementById('highest-price');
const domLow = document.getElementById('lowest-price');
const domOpen = document.getElementById('open-price');
const domClose = document.getElementById('current-price');

const canvas = iframe.getElementById('canvas');
const ctx = canvas.getContext('2d');
const candleWidth = 5;
const candleSpacing = 7;

let width = canvas.width;
let height = canvas.height;

let priceArray, px, high, low;

const grid = new class {
 constructor() {
  this.saved = document.createElement('canvas');
  this.saved.width = width;
  this.saved.height = height;
  this.ctx = this.saved.getContext('2d');
  
  const patternCanvas = document.createElement("canvas");
  const pCtx = patternCanvas.getContext("2d");
  
  patternCanvas.width = 30;
  patternCanvas.height = 30;
  
  pCtx.setLineDash([3, 8]);
  pCtx.strokeWidth = 2;
  pCtx.strokeStyle = 'darkslategray';
  pCtx.moveTo(0, 2);
  pCtx.lineTo(width, 0);
  
  pCtx.moveTo(0, 0);
  pCtx.lineTo(0, height);
  
  pCtx.stroke();
  
  this.pattern = this.ctx.createPattern(patternCanvas, "repeat");
  
  this.drawGrid();
 }
 drawGrid() {
  this.ctx.fillStyle = this.pattern;
  this.ctx.fillRect(0, 0, width, height);
  
  /* Draw a white line over the below black container */
  this.ctx.beginPath();
  this.ctx.strokeStyle = "white";
  drawLine(0, height - 25, width, height - 25, this.ctx);
  /* Draw a black container to cover grids and show time clearly */
  this.ctx.fillStyle = 'black';
  this.ctx.fillRect(0, height - 25, width, 25);
  this.ctx.fill();
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

candles = new class CandleSticks {
 constructor() {
  this.bullColor = 'white';
  this.bearColor = 'red';
  this.chartImg = document.createElement('canvas');
  this.chartImg.width = width;
  this.chartImg.height = height;
 }
 drawCandles(rates, highs, lows) {
  let y1, y2, x1 = 40,
   x2 = 41,
   cd = 1;
  
  ctx.clearRect(0, 0, width, height);
  ctx.drawImage(grid.saved, 0, 0);
  
  // Get highest and lowest point in price
  high = Math.max.apply(null, rates.map(a => a.high)) + 10;
  low = Math.min.apply(null, rates.map(a => a.low)) + 5;
  px = ((height / 1.2) / (high - low));
  
  ctx.textAlign = 'center';
  let ratesLen = rates.length;
  
  for (let candle of rates) {
   let info = this.getInfo(candle);
   
   y1 = info.type ? (high - candle.close) * px : (high - candle.open) * px;
   y2 = info.type ? ((high - candle.open) * px) - y1 : ((high - candle.close) * px) - y1;
   
   ctx.beginPath();
   ctx.lineWidth = 1;
   ctx.strokeStyle = info.color;
   drawLine(x1 + (candleWidth / 2), (high - candle.high) * px, x1 + (candleWidth / 2), (high - candle.low) * px);
   ctx.fillStyle = info.color;
   ctx.fillRect(x2, y1, 3, y2);
   
   // Number candles
   ctx.strokeStyle = 'white';
   ctx.fillStyle = 'white';
   ctx.fillText(ratesLen - cd, x1 + (candleWidth / 2), ((high - candle.low) * px) + 10);
   /*
      // Swings
      let obHigh = highs.find(e => e.Index == cd);
      let obLow = lows.find(e => e.Index == cd);
      
      if (obHigh) ctx.fillText('SWH', x1 + (candleWidth / 2), (high - candle.high) * px);
      if (obLow) ctx.fillText('SWL', x1 + (candleWidth / 2), ((high - candle.low) * px) + 20);
      
      if(obHigh) {
       ctx.strokeStyle = 'rgba(1, 4, 37, 0.7)';
       ctx.fillStyle = 'rgba(1, 4, 120, 0.7)';
       y1 = (high - obHigh.UpperBound) * px;
       y2 = ((high - obHigh.LowerBound) * px) - y1;
       
       ctx.fillRect(0, y1, x2, y2);
      }
      
      if(obLow) {
        ctx.strokeStyle = 'rgba(157, 171, 0, 0.7)';
       ctx.fillStyle = 'rgba(157, 171, 0, 0.7)';
       y1 = (high - obLow.UpperBound) * px;
       y2 = ((high - obLow.LowerBound) * px) - y1;
       
       ctx.fillRect(0, y1, x2, y2);
      }
      */
   x1 += candleSpacing;
   x2 += candleSpacing;
   
   cd++;
  }
 }
 getInfo(candle) {
  let type = candle.open < candle.close;
  return {
   color: type ? this.bullColor : this.bearColor,
   type
  }
 }
}

candles.drawCandles(candlesArr /*, highs, lows*/ );