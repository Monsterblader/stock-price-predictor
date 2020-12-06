const resizeViews = {
  active: {
    'border-radius': '0px',
    height: '1080px',
    left: '0px',
    top: '0px',
    width: '1920px',
  },

  box: {
    'background-color': '#fff',
    'border-radius': '5px',
    height: '108px',
    position: 'absolute',
    width: '192px',
  },

  one: {
    left: '236px',
    top: '161px',
  },

  two: {
    left: '453px',
    top: '353px',
  },

  three: {
    left: '642px',
    top: '542px',
  },

  four: {
    left: '832px',
    top: '275px',
  },

  five: {
    left: '1114px',
    top: '117px',
  },

  six: {
    left: '976px',
    top: '457px',
  },

  seven: {
    left: '1403px',
    top: '628px',
  },

  expand: source => {
    const target = d3.select('.box.' + source);
    const cover = d3.select('.box.' + source + ' .cover');

    target.style('z-index', 10);
    // zoom cover to left corner
    target.classed('active', true)
      .transition()
      .duration(1000)
      .ease(d3.easeExpInOut)
      .style('border-radius', '0px')
      .style('left', '0px')
      .style('top', '0px')
      .style('width', '1920px')
      .style('height', '1080px')
      .style('opacity', 1);

    cover.transition()
      .duration(1000)
      .ease(d3.easeExpInOut)
      .style('left', '0%')
      .style('top', '0%')
      .style('width', '0px')
      .style('height', '0px')
      .style('font-size', '0px');
  },

  restore: source => {
    d3.select('.active .cover')
      .transition()
      .duration(1000)
      .ease(d3.easeExpInOut)
      .style('top', '0px')
      .style('height', '108px')
      .style('width', '192px')
      .style('font-size', '24px');

    d3.select('.active')
      .classed('active', false)
      .transition()
      .duration(1000)
      .ease(d3.easeExpInOut)
      .style('border-radius', '10px')
      .style('left', resizeViews[source].left)
      .style('top', resizeViews[source].top)
      .style('width', '192px')
      .style('height', '108px')
      .style('opacity', 0.9)
      .style('z-index', 1);
  },

  moveView: (left = 0, top = 0) => {
    d3.select('.map')
      .transition()
      .duration(1000)
      .style('left', left + "px")
      .style('top', top + "px");
  },
}

const getChart = () => {
  const cb = data => {
    const result = JSON.parse(JSON.parse(data).df);
    const test = Object.values(result.test);
    const pred = Object.values(result.pred);
    const chart1 = [];
    const chart2 = [];

    for (let i = 0, l = test.length; i < l; i += 1) {
      chart1[i] = { index: i, val: test[i] };
      chart2[i] = { index: i, val: pred[i] };
    }
    d3.select('#my_dataviz')._groups[0][0].innerHTML = "";

    //No.1 define the svg
    const graphWidth = 600,
      graphHeight = 450,
      margin = {
        top: 30,
        right: 10,
        bottom: 30,
        left: 85
      },
      totalWidth = graphWidth + margin.left + margin.right,
      totalHeight = graphHeight + margin.top + margin.bottom,
      svg = d3
        .select("#my_dataviz")
        .append("svg")
        .attr("width", totalWidth)
        .attr("height", totalHeight),

      //No.2 define mainGraph
      mainGraph = svg
        .append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")"),

      //No.3 define axes
      categoriesNames = chart1.map((d) => d.index),
      xScale = d3
        .scalePoint()
        .domain(categoriesNames)
        .range([0, graphWidth]), // scalepoint make the axis starts with value compared with scaleBand
      yScale = d3
        .scaleLinear()
        .range([graphHeight, 0])
        .domain([0, d3.max(chart1, data => data.val)]), //* If an arrow function is simply returning a single line of code, you can omit the statement brackets and the return keyword

      //No.5 make lines
      line = d3
        .line()
        .x(function (d) {
          return xScale(d.index);
        }) // set the x values for the line generator
        .y(function (d) {
          return yScale(d.val);
        }); // set the y values for the line generator
        // .curve(d3.curveMonotoneX); // apply smoothing to the line

    //No.4 set axises
    mainGraph.append("g")
      .attr("class", "x axis")
      .attr("transform", "translate(0," + graphHeight + ")")
      .call(d3.axisBottom(xScale));

    mainGraph.append("g")
      .attr("class", "y axis")
      .call(d3.axisLeft(yScale));

    mainGraph.append("path")
      .datum(chart1) // 10. Binds data to the line
      .attr("class", "line") // Assign a class for styling
      .attr("d", line); // 11. Calls the line generator

    mainGraph.append("path")
      .datum(chart2) // 10. Binds data to the line
      .attr("class", "line2") // Assign a class for styling
      .attr("d", line); // 11. Calls the line generator

    const tix = $('#my_dataviz .x.axis .tick');

    for (let i = 0, l = tix.length; i < l; i += 1) {
      if (i % 100 !== 0) {
        tix[i].innerHTML = '';
      }
    }

    $('#get-prediction')[0].innerText = 'Go!';
  } // end cb

  const getParameters = () => {
    const params = {
      ticker: $('#ticker-symbol')[0].value,
    };
    const inputs = $('.box.five .checkbox-group');

    for (let i = 0, l = inputs.length; i < l; i += 1) {
      const node = inputs[i].firstElementChild;
      if (node.checked) {
        params[node.id] = true;
      }
    }

    return params;
  }

  $('#get-prediction')[0].innerText = 'Loading...';
  $.ajax({
    url: "getprediction",
    type: "get",
    data: getParameters(),
    success: cb,
  });
} // end getChart

const toggleIndicators = className => {
  const els = $(`.box.five .${className} input`);

  for (let i = 0, l = els.length; i < l; i += 1) {
    els[i].disabled = !els[i].disabled;
  }
}

const checkKey = e => {
  const err = e || window.event;
  const getBox = () => {
    let className = '';

    return d3.select('.active')._groups[0][0].classList[1];
  }

  if (e.keyCode == '27') {
    resizeViews.restore(getBox());
  }
}

toggleIndicators('not-implemented');
toggleIndicators('data-frame');
document.onkeyup = checkKey;

