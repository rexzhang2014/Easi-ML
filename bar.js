function fetchData() {
    // 通过 setTimeout 模拟异步加载
	
	$.getJSON('ajax/test.json', function(data) {
  var items = [];
 
  $.each(data, function(key, val) {
    items.push('<li id="' + key + '">' + val + '</li>');
  });
 
  $('<ul/>', {
    'class': 'my-new-list',
    html: items.join('')
  }).appendTo('body');
});
	/*return {
            categories: ["衬衫","羊毛衫","雪纺衫","裤子","高跟鞋","袜子"],
            data: [5, 20, 36, 10, 10, 20]
        };	*/
};

// 初始 option
var option = {
    title: {
        text: '异步数据加载示例'
    },
    tooltip: {},
    legend: {
        data:['销量']
    },
    xAxis: {
        data: []
    },
    yAxis: {},
    series: [{
        name: '销量',
        type: 'bar',
        data: []
    }]
};



var myChart = echarts.init(document.getElementById('chart01'));
// 使用刚指定的配置项和数据显示图表。
data = fetchData() ;
myChart.setOption(option); 

myChart.setOption({
        xAxis: {
            data: data.categories
        },
        series: [{
            // 根据名字对应到相应的系列
            name: '销量',
            data: data.data
        }]
    });
