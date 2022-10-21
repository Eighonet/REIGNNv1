import React, { Component } from 'react';
import './css/Stats.css';
import { Bar, Radar } from "react-chartjs-2";
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    BarElement,
    RadialLinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend,
} from 'chart.js';

ChartJS.register(
    BarElement,
    RadialLinearScale,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend
);

const options = {
    maintainAspectRatio: false	// Don't maintain w/h ratio
}

const options_r = {
    plugins: {
        legend: {
            display: false,
        },
    },

};

const options_b = {
    plugins: {
        legend: {
            display: false,
        },
    },
};



class Stats extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            number: 0,
            sjr: 0,
            if: 0,
            hi: 0,
            interests: [],
            data_bar: {
                labels: ['0.2<', '0.2', '0.4', '0.6', '0.8', '1.0', '1.2', '1.4', '1.6', '1.8', '2.0', '>2.0'],
                datasets: [
                    {
                        label: "SJR",
                        backgroundColor: '#EC932F',
                        borderColor: 'rgba(255,99,132,1)',
                        borderWidth: 1,
                        hoverBackgroundColor: 'rgba(255,99,132,0.4)',
                        hoverBorderColor: 'rgba(255,99,132,1)',
                        data: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    },
                    {
                        label: "Impact factor",
                        backgroundColor: 'green',
                        borderColor: 'rgba(255,99,132,1)',
                        borderWidth: 1,
                        hoverBackgroundColor: 'rgba(255,99,132,0.4)',
                        hoverBorderColor: 'rgba(255,99,132,1)',
                        data: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                    },

                ]
            },
            data_radar: {
                labels: ['Medicine', 'CS', 'Biology', 'Physics', 'Art', 'Economics'],
                datasets: [
                    {
                        data: [0, 0, 0, 0, 0, 0],
                        backgroundColor: 'rgba(255, 99, 132, 0.2)',
                        borderColor: 'rgba(255, 99, 132, 1)',
                        borderWidth: 1,
                    },
                ],
            },
            cats: ['Art',
                'Biology',
                'Business',
                'Chemistry',
                'Computer Science',
                'Economics',
                'Engineering',
                'Environmental Science',
                'Geography',
                'Geology',
                'History',
                'Materials Science',
                'Mathematics',
                'Medicine',
                'Philosophy',
                'Physics',
                'Political Science',
                'Psychology',
                'Sociology']
        }
    }

    onTestClick() {
        this.setState({
            data_radar: {
                labels: ['Thing 1', 'Thing 2', 'Thing 3', 'Thing 4', 'Thing 5', 'Thing 6'],
                datasets: [
                    {
                        label: '# of Votes',
                        data: [0, 0, 0, 0, 0, 0],
                        backgroundColor: 'rgba(255, 99, 132, 0.2)',
                        borderColor: 'rgba(255, 99, 132, 1)',
                        borderWidth: 1,
                    },
                ],
            }
        });
    }

    profile_sort(input) {
        let sortable = [];
        for (let part in input) {
            sortable.push([part, input[part]]);
        }

        sortable.sort(function(a, b) {
            return b[1] - a[1];
        });
        return sortable
    }

    transpose(matrix) {
        return matrix[0].map((col, i) => matrix.map(row => row[i]));
    }

    avg(array) {
        const sum = array.reduce((a, b) => a + b, 0);
        const avg = (sum / array.length) || 0;
        return avg
    }

    sum(array) {
        const sum = array.reduce((a, b) => a + b, 0);
        return sum
    }

    componentWillReceiveProps(nextProps) {
        let feature_vectors = [], interests = nextProps.interests
        if (nextProps.interests !== this.state.interests) {
            console.log("Interests", nextProps.interests)
            this.setState({ interests: nextProps.interests });
            for (let i = 0; i < interests.data.length; i++) {
                feature_vectors.push(interests.data[i].feature_vector.map(Number))
            }
            let sum = (r, a) => r.map((b, i) => a[i] + b);
            console.log("interests", (feature_vectors).reduce(sum));
            let result = (feature_vectors).reduce(sum)
            let dict =  new Object()
            for (let i = 0; i < this.state.cats.length; i++) {
                dict[this.state.cats[i]] = result[i];
            }
            console.log(dict, this.transpose(this.profile_sort(dict).slice(0, 6)))
            result = this.transpose(this.profile_sort(dict).slice(0, 6))
            this.setState({ data_radar: {
                    labels: result[0],
                    datasets: [
                        {
                            data: result[1],
                            backgroundColor: 'rgba(255, 99, 132, 0.2)',
                            borderColor: 'rgba(255, 99, 132, 1)',
                            borderWidth: 1,
                        },
                    ],
                }
            })

            let graph_features = []

            let borders = [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 10]
            let if_dist = Array(borders.length).fill(0), sjr_dist = Array(borders.length).fill(0)

            let gft = this.transpose(graph_features)
            let sjr = gft[1].sort(function(a, b) {
                return a - b;
            }), ifact = gft[2].sort(function(a, b) {
                return a - b;
            })

            for (let j=0; j < ifact.length; j++) {
                for (let i = 0; i < borders.length - 1; i++) {
                    if (ifact[j] > borders[i] && ifact[j] < borders[i+1]) {
                        if_dist[i] += 1
                    }
                }
            }

            for (let j=0; j < sjr.length; j++) {
                for (let i = 0; i < borders.length - 1; i++) {
                    if (sjr[j] > borders[i] && sjr[j] < borders[i+1]) {
                        sjr_dist[i] += 1
                    }
                }
            }

            if_dist = if_dist.map(function(v){
                return v/ifact.length;
            });

            sjr_dist = sjr_dist.map(function(v){
                return v/sjr.length;
            });

            console.log(if_dist, sjr_dist)

            this.setState({number: this.sum(gft[0]),
                sjr: this.avg(gft[1]),
                if: this.avg(gft[2]),
                hi: this.avg(gft[3]),
                data_bar: {
                    labels: ['0.2<', '0.2', '0.4', '0.6', '0.8', '1.0', '1.2', '1.4', '1.6', '1.8', '2.0', '>2.0'],
                    datasets: [
                        {
                            label: "SJR",
                            backgroundColor: '#EC932F',
                            borderColor: 'rgba(255,99,132,1)',
                            borderWidth: 1,
                            hoverBackgroundColor: 'rgba(255,99,132,0.4)',
                            hoverBorderColor: 'rgba(255,99,132,1)',
                            data: sjr_dist
                        },
                        {
                            label: "Impact factor",
                            backgroundColor: 'green',
                            borderColor: 'rgba(255,99,132,1)',
                            borderWidth: 1,
                            hoverBackgroundColor: 'rgba(255,99,132,0.4)',
                            hoverBorderColor: 'rgba(255,99,132,1)',
                            data: if_dist
                        },

                    ]
                },
            })

            this.setState({graph_features: graph_features})

        }
    }

    render() {
        let interests = this.props.interests
        return(
            <div className="panels">
                    <div className="panel summary">
                        <div className="inner">
                            <div className="stats_title">Team statistics</div>
                            <div className="stats">
                                <div className="stat pubs_number">
                                    <div className="stat_title">Total publications </div>
                                    <div className="value pubs_number_value">{this.state.number}</div>
                                </div>
                                <div className="stat if">
                                    <div className="stat_title">Average impact factor </div>
                                    <div className="value if_value">{this.state.if}</div>
                                </div>
                                <div className="stat sjr">
                                    <div className="stat_title">Average SJR </div>
                                    <div className="value sjr_value">{this.state.sjr}</div>
                                </div>
                                <div className="stat hi">
                                    <div className="stat_title">Average h-index </div>
                                    <div className="value hi_value">{this.state.hi}</div>
                                </div>
                            </div>
                        </div>
                    </div>

                <div className="panel">
                    <div className="radar">
                        <div className="chart_title">Team specialization</div>
                        <Radar data={this.state.data_radar}
                               options={options_r}
                               width={6}
                               height={4}/>
                    </div>
                </div>

                <div className="panel">
                    <div className="distribution">
                        <div className="chart_title">Publications quality distribution</div>
                        <Bar data={this.state.data_bar}
                             options={options_b}
                             width={6}
                             height={4}/>
                    </div>
                </div>

            </div>
        )
    }
}

export default Stats;