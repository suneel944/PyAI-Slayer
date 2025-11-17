/**
 * PyAI-Slayer Dashboard - Advanced Charts (Phase 2)
 * ApexCharts integration for interactive visualizations
 */

let apexCharts = {};

// ===== SPARKLINES FOR METRIC CARDS =====
function createSparkline(elementId, data, color) {
    const options = {
        series: [{
            data: data
        }],
        chart: {
            type: 'line',
            width: '100%',
            height: 60,
            sparkline: {
                enabled: true
            },
            animations: {
                enabled: true,
                easing: 'easeinout',
                speed: 800
            }
        },
        stroke: {
            curve: 'smooth',
            width: 2
        },
        colors: [color],
        tooltip: {
            enabled: true,
            theme: currentTheme,
            y: {
                formatter: function (val) {
                    return val.toFixed(0);
                }
            }
        }
    };

    if (apexCharts[elementId]) {
        apexCharts[elementId].destroy();
    }

    const element = document.getElementById(elementId);
    if (element) {
        apexCharts[elementId] = new ApexCharts(element, options);
        apexCharts[elementId].render();
    }
}

// ===== ENHANCED DONUT CHART WITH DRILL-DOWN =====
function createInteractiveDonutChart(elementId, data) {
    const isDark = currentTheme === 'dark';
    // Get the global font family from CSS or use system fonts
    const globalFont = getComputedStyle(document.documentElement).getPropertyValue('--font-family').trim() || 
                       'Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif';
    
    const options = {
        series: data.values,
        chart: {
            type: 'donut',
            height: 400,
            background: 'transparent',
            fontFamily: globalFont,
            animations: {
                enabled: true,
                easing: 'easeinout',
                speed: 1000,
                animateGradually: {
                    enabled: true,
                    delay: 200
                },
                dynamicAnimation: {
                    enabled: true,
                    speed: 400
                }
            },
            events: {
                dataPointSelection: function(event, chartContext, config) {
                    const label = data.labels[config.dataPointIndex];
                    showToast('Filter Applied', `Showing ${label} tests`, 'info');
                }
            }
        },
        labels: data.labels,
        colors: ['#10b981', '#ef4444', '#f59e0b'],
        theme: {
            mode: isDark ? 'dark' : 'light'
        },
        legend: {
            show: true,
            position: 'bottom',
            horizontalAlign: 'center',
            fontSize: '14px',
            fontFamily: globalFont,
            fontWeight: 500,
            labels: {
                colors: isDark ? '#f1f5f9' : '#0f172a',
                useSeriesColors: false
            },
            markers: {
                width: 14,
                height: 14,
                radius: 3,
                offsetX: -4
            },
            itemMargin: {
                horizontal: 16,
                vertical: 8
            }
        },
        dataLabels: {
            enabled: true,
            formatter: function(val, opts) {
                return opts.w.config.series[opts.seriesIndex]
            },
            style: {
                fontSize: '16px',
                fontFamily: globalFont,
                fontWeight: '700',
                colors: ['#fff']
            },
            dropShadow: {
                enabled: true,
                top: 1,
                left: 1,
                blur: 2,
                opacity: 0.5
            },
            offsetY: 0
        },
        plotOptions: {
            pie: {
                startAngle: 0,
                endAngle: 360,
                expandOnClick: true,
                offsetX: 0,
                offsetY: 0,
                customScale: 1,
                dataLabels: {
                    offset: 0,
                    minAngleToShowLabel: 10
                },
                donut: {
                    size: '65%',
                    background: 'transparent',
                    labels: {
                        show: true,
                        name: {
                            show: true,
                            fontSize: '18px',
                            fontFamily: globalFont,
                            fontWeight: 600,
                            color: isDark ? '#f1f5f9' : '#0f172a',
                            offsetY: -10
                        },
                        value: {
                            show: true,
                            fontSize: '36px',
                            fontFamily: globalFont,
                            fontWeight: 'bold',
                            color: isDark ? '#3b82f6' : '#2563eb',
                            offsetY: 10,
                            formatter: function (val) {
                                return val
                            }
                        },
                        total: {
                            show: true,
                            showAlways: true,
                            label: 'Total Tests',
                            fontSize: '16px',
                            fontFamily: globalFont,
                            fontWeight: 600,
                            color: isDark ? '#94a3b8' : '#64748b',
                            formatter: function (w) {
                                const total = w.globals.seriesTotals.reduce((a, b) => a + b, 0);
                                return total;
                            }
                        }
                    }
                }
            }
        },
        states: {
            hover: {
                filter: {
                    type: 'lighten',
                    value: 0.1
                }
            },
            active: {
                allowMultipleDataPointsSelection: false,
                filter: {
                    type: 'darken',
                    value: 0.2
                }
            }
        },
        stroke: {
            show: true,
            curve: 'smooth',
            lineCap: 'round',
            colors: [isDark ? '#1e293b' : '#ffffff'],
            width: 3
        },
        responsive: [{
            breakpoint: 480,
            options: {
                chart: {
                    height: 320
                },
                legend: {
                    position: 'bottom',
                    fontSize: '12px'
                },
                plotOptions: {
                    pie: {
                        donut: {
                            labels: {
                                name: {
                                    fontSize: '14px'
                                },
                                value: {
                                    fontSize: '28px'
                                },
                                total: {
                                    fontSize: '14px'
                                }
                            }
                        }
                    }
                }
            }
        }],
        tooltip: {
            enabled: true,
            theme: isDark ? 'dark' : 'light',
            style: {
                fontSize: '14px',
                fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto'
            },
            y: {
                formatter: function(val) {
                    return val + " tests";
                },
                title: {
                    formatter: function(seriesName) {
                        return seriesName + ': ';
                    }
                }
            }
        }
    };

    if (apexCharts[elementId]) {
        apexCharts[elementId].destroy();
    }

    const element = document.getElementById(elementId);
    if (element) {
        apexCharts[elementId] = new ApexCharts(element, options);
        apexCharts[elementId].render();
    }
}

// ===== REAL-TIME LINE CHART =====
function createRealTimeLineChart(elementId, data) {
    const isDark = currentTheme === 'dark';
    
    const options = {
        series: [{
            name: 'Pass Rate',
            data: data.passRate || []
        }, {
            name: 'Failed Tests',
            data: data.failedTests || []
        }],
        chart: {
            type: 'area',
            height: 350,
            background: 'transparent',
            animations: {
                enabled: true,
                easing: 'linear',
                dynamicAnimation: {
                    speed: 1000
                }
            },
            toolbar: {
                show: true,
                tools: {
                    download: true,
                    selection: true,
                    zoom: true,
                    zoomin: true,
                    zoomout: true,
                    pan: true,
                    reset: true
                }
            },
            zoom: {
                enabled: true,
                type: 'x',
                autoScaleYaxis: true
            }
        },
        dataLabels: {
            enabled: false
        },
        stroke: {
            curve: 'smooth',
            width: 2
        },
        fill: {
            type: 'gradient',
            gradient: {
                shadeIntensity: 1,
                opacityFrom: 0.7,
                opacityTo: 0.3,
                stops: [0, 90, 100]
            }
        },
        colors: ['#10b981', '#ef4444'],
        theme: {
            mode: isDark ? 'dark' : 'light'
        },
        xaxis: {
            type: 'datetime',
            labels: {
                style: {
                    colors: isDark ? '#94a3b8' : '#475569'
                }
            }
        },
        yaxis: {
            labels: {
                style: {
                    colors: isDark ? '#94a3b8' : '#475569'
                },
                formatter: function(val) {
                    return val ? val.toFixed(0) : '0';
                }
            }
        },
        legend: {
            labels: {
                colors: isDark ? '#f1f5f9' : '#0f172a'
            }
        },
        grid: {
            borderColor: isDark ? '#334155' : '#e2e8f0',
            strokeDashArray: 4
        },
        tooltip: {
            theme: isDark ? 'dark' : 'light',
            x: {
                format: 'dd MMM yyyy HH:mm'
            }
        }
    };

    if (apexCharts[elementId]) {
        apexCharts[elementId].destroy();
    }

    const element = document.getElementById(elementId);
    if (element) {
        apexCharts[elementId] = new ApexCharts(element, options);
        apexCharts[elementId].render();
    }
}

// ===== RADIAL BAR CHART FOR METRICS =====
function createRadialChart(elementId, value, label, color) {
    const isDark = currentTheme === 'dark';
    
    const options = {
        series: [value],
        chart: {
            type: 'radialBar',
            height: 200,
            background: 'transparent',
            animations: {
                enabled: true,
                easing: 'easeinout',
                speed: 800
            }
        },
        plotOptions: {
            radialBar: {
                hollow: {
                    size: '60%'
                },
                track: {
                    background: isDark ? '#334155' : '#e2e8f0',
                    strokeWidth: '100%'
                },
                dataLabels: {
                    show: true,
                    name: {
                        show: true,
                        fontSize: '12px',
                        color: isDark ? '#94a3b8' : '#475569',
                        offsetY: 20
                    },
                    value: {
                        show: true,
                        fontSize: '24px',
                        fontWeight: 'bold',
                        color: isDark ? '#f1f5f9' : '#0f172a',
                        offsetY: -10,
                        formatter: function(val) {
                            return val + '%'
                        }
                    }
                }
            }
        },
        colors: [color],
        labels: [label],
        theme: {
            mode: isDark ? 'dark' : 'light'
        }
    };

    if (apexCharts[elementId]) {
        apexCharts[elementId].destroy();
    }

    const element = document.getElementById(elementId);
    if (element) {
        apexCharts[elementId] = new ApexCharts(element, options);
        apexCharts[elementId].render();
    }
}

// ===== BAR CHART WITH DRILL-DOWN =====
function createInteractiveBarChart(elementId, data) {
    const isDark = currentTheme === 'dark';
    
    const options = {
        series: [{
            name: 'Count',
            data: data.values || []
        }],
        chart: {
            type: 'bar',
            height: 350,
            background: 'transparent',
            toolbar: {
                show: true
            },
            animations: {
                enabled: true,
                easing: 'easeinout',
                speed: 800
            },
            events: {
                dataPointSelection: function(event, chartContext, config) {
                    const category = data.labels[config.dataPointIndex];
                    showToast('Category Selected', `Viewing ${category} details`, 'info');
                }
            }
        },
        plotOptions: {
            bar: {
                borderRadius: 8,
                horizontal: false,
                columnWidth: '60%',
                dataLabels: {
                    position: 'top'
                }
            }
        },
        dataLabels: {
            enabled: true,
            offsetY: -20,
            style: {
                fontSize: '12px',
                colors: [isDark ? '#f1f5f9' : '#0f172a']
            }
        },
        colors: ['#ef4444'],
        xaxis: {
            categories: data.labels || [],
            labels: {
                style: {
                    colors: isDark ? '#94a3b8' : '#475569',
                    fontSize: '12px'
                }
            }
        },
        yaxis: {
            labels: {
                style: {
                    colors: isDark ? '#94a3b8' : '#475569'
                },
                formatter: function(val) {
                    return val ? val.toFixed(0) : '0';
                }
            }
        },
        grid: {
            borderColor: isDark ? '#334155' : '#e2e8f0',
            strokeDashArray: 4
        },
        theme: {
            mode: isDark ? 'dark' : 'light'
        },
        tooltip: {
            theme: isDark ? 'dark' : 'light',
            y: {
                formatter: function(val) {
                    return val + " failures"
                }
            }
        }
    };

    if (apexCharts[elementId]) {
        apexCharts[elementId].destroy();
    }

    const element = document.getElementById(elementId);
    if (element) {
        apexCharts[elementId] = new ApexCharts(element, options);
        apexCharts[elementId].render();
    }
}

// ===== MULTI-LINE CHART FOR VALIDATION METRICS =====
function createValidationMetricsChart(elementId, data) {
    const isDark = currentTheme === 'dark';
    
    const options = {
        series: data.series || [],
        chart: {
            type: 'line',
            height: 350,
            background: 'transparent',
            toolbar: {
                show: true
            },
            animations: {
                enabled: true,
                easing: 'easeinout',
                speed: 800
            }
        },
        stroke: {
            width: 3,
            curve: 'smooth'
        },
        markers: {
            size: 5,
            hover: {
                size: 7
            }
        },
        colors: ['#3b82f6', '#10b981', '#f59e0b', '#06b6d4'],
        xaxis: {
            categories: data.categories || [],
            labels: {
                style: {
                    colors: isDark ? '#94a3b8' : '#475569'
                }
            }
        },
        yaxis: {
            min: 0,
            max: 100,
            labels: {
                style: {
                    colors: isDark ? '#94a3b8' : '#475569'
                },
                formatter: function(val) {
                    return val ? val.toFixed(0) + '%' : '0%';
                }
            }
        },
        legend: {
            labels: {
                colors: isDark ? '#f1f5f9' : '#0f172a'
            }
        },
        grid: {
            borderColor: isDark ? '#334155' : '#e2e8f0',
            strokeDashArray: 4
        },
        theme: {
            mode: isDark ? 'dark' : 'light'
        },
        tooltip: {
            theme: isDark ? 'dark' : 'light',
            shared: true,
            intersect: false,
            y: {
                formatter: function(val) {
                    return val ? val.toFixed(1) + '%' : '0%';
                }
            }
        }
    };

    if (apexCharts[elementId]) {
        apexCharts[elementId].destroy();
    }

    const element = document.getElementById(elementId);
    if (element) {
        apexCharts[elementId] = new ApexCharts(element, options);
        apexCharts[elementId].render();
    }
}

// ===== DESTROY ALL APEX CHARTS (for theme changes) =====
function destroyAllApexCharts() {
    Object.keys(apexCharts).forEach(key => {
        if (apexCharts[key]) {
            apexCharts[key].destroy();
        }
    });
    apexCharts = {};
}

// ===== UPDATE CHARTS ON THEME CHANGE =====
function updateChartsForTheme() {
    destroyAllApexCharts();
    // Charts will be recreated when data is loaded
}
