const getChart = () => {
  const cb = data => {
    const dollars = new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2
    });
    const result = JSON.parse(data),
      test = Object.values(result.test),
      pred = Object.values(result.pred),
      chart1 = [],
      chart2 = [];

    for (let i = 0, l = test.length; i < l; i += 1) {
      chart1[i] = { index: i, val: test[i] };
      chart2[i] = { index: i, val: pred[i] };
    }
    d3.select('#my_dataviz')._groups[0][0].innerHTML = "";

    //No.1 define the svg
    const graphWidth = 1000,
      graphHeight = 600,
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

    $('.box.five .metrics')[0].style.visibility = 'initial';
    $('.box.five .company-name')[0].style.visibility = 'initial';
    $('.box.five .company-name')[0].innerText = result.company['0'];
    $('.box.five .metrics .days')[0].innerText = 10;
    $('.box.five .metrics .mae')[0].innerText = result.mean_absolute_error['0'].toFixed(3);
    $('.box.five .metrics .accuracy')[0].innerText = result.accuracy['0'].toFixed(3);
    $('.box.five .metrics .price')[0].innerText = dollars.format(result.future_price['0']);
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

toggleIndicators('not-implemented');
toggleIndicators('data-frame');