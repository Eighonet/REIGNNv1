import './css/Graph.css';
import React from "react";
import ReactDOM from "react-dom";
import Graph from "react-graph-vis";
import cloneDeep from "lodash/cloneDeep";
import {useState} from "react-js-dialog-box";

const me = "https://raw.githubusercontent.com/Eighonet/REIGNN_demo/main/front/user_icon.png";

class Network extends React.Component {
    constructor(props) {
        super(props);

        this.graph =  {
            nodes: [],
            edges: []
        }
        this.state = {
            panel_trigger_1: 0,
            panel_trigger_2: 2,
            panel_hidden: "visible",
            placedDiv: {
                top: "15px",
                left: "340px",
            },
            network: null,
            received_graph: [],
            node_ids: [],
            edges: [],
            stats: {
                number: 0,
                sjr: 0,
                if: 0,
                hi: 0,
            },
            graph_features: []
        }
    }
    componentDidMount(){
        this.state = { node_ids: [], edges: []};
    }

    showWhereClicked = (e) => {
        console.log(`you have clicked X:${e.screenX} Y:${e.screenY}`);
        console.log(this.state)
/*        if (this.state.panel_trigger_1 % 2 != 0) {
            this.setState({
                panel_trigger_1: this.state.panel_trigger_1 + 1
            })
        }
 */
        /*
        if (this.state.panel_trigger_1 == this.state.panel_trigger_2) {
            let new_state = this.state.panel_trigger_2 + 2
            this.setState({
                placedDiv: {
                    visibility: "visible",
                    top: e.screenY - 100 + 'px',
                    left: e.screenX - 101 + 'px'
                },
                panel_trigger_2: new_state
            })
        } else {
            this.setState({
                placedDiv: {
                    visibility: "hidden",
                    top: e.screenY - 100 + 'px',
                    left: e.screenX - 101 + 'px'
                }
            })
        }
        */

//        console.log(this.state)
    }

    componentWillReceiveProps(nextProps) {
        console.log("Difference", nextProps.graph, this.state.received_graph, nextProps.graph !== this.state.received_graph)
        console.log("Difference",nextProps.node_ids, this.state.node_ids)
        if (nextProps.node_ids !== this.state.node_ids &&
            nextProps.graph !== this.state.received_graph
//            && typeof this.state.received_graph !== "undefined"
        ) {

            this.setState({ node_ids: nextProps.node_ids });
            let graphData = this.graph
            console.log("State graph", this.graph)
            let received_graph = nextProps.graph.author_edgelist.data
            let received_graph_hi = nextProps.graph.z_hi, received_graph_sjr = nextProps.graph.z_sjr,
                received_graph_numb = nextProps.graph.z_numb, received_graph_ifact = nextProps.graph.z_ifact

            console.log("Received graph", nextProps.graph.author_edgelist.data)
            console.log("Received graph", nextProps.node_ids)
            let newGraph = cloneDeep(graphData);
            let nodes = [], edges = []
            for (let i=0; i<nextProps.node_ids.length; i++) {
                console.log(nextProps.node_ids[i]["id"])
                nodes.push({ id: nextProps.node_ids[i]["id"], label: nextProps.node_ids[i]["name"], image: me})
            }

            for (let i=0; i<received_graph.length; i++) {
                edges.push({from: received_graph[i]["author_a"], to: received_graph[i]["author_b"]})
            }

            newGraph.nodes = nodes
            newGraph.edges = edges
            this.setState({graph: newGraph});

            let graph_features = new Object()
            for (let i=0; i<received_graph.length; i++) {
                graph_features[[received_graph[i]["author_a"], received_graph[i]["author_b"]]] = [Math.abs(received_graph_numb[i]),
                    Math.abs(received_graph_ifact[i]),
                    Math.abs(received_graph_sjr[i]),
                    Math.abs(received_graph_hi[i])]
            }

            this.setState({graph_features: graph_features})

        } else {
            let graphData = this.graph
            let newGraph = cloneDeep(graphData);
            let nodes = [], edges = []
            this.setState({graph: newGraph});

            let graph_features = new Object()
            this.setState({graph_features: graph_features})
        }
    }

    render() {
        let node_ids = this.props.node_ids
        let received_graph = this.props.graph
        let nodes = [], edges = []

        if (node_ids.length != 0) {
            let obj = this
            let graph = this.state.graph
            let network_a = this.state.network

            const options = {
                layout: {
                    hierarchical: false
                },
                nodes: {
                    shape: "circularImage",
                    size: 30,
                    chosen: false,
                    color: {
                        background: "white"
                    },
                    borderWidth: 0,
                },
                edges: {
                    color: "#000000",
                    arrows: {
                        to: false,
                        from: false
                    }
                },
                height: "1000px"
            };

            const events = {
                select: function(event) {
                    obj.setState({network: network_a})
                    network_a = obj.state.network

                    var {nodes, edges} = event;

                    console.log("Event occured")
                    console.log(event)

                    if (nodes.length > 0) {
//                        console.log("node")
                        let node = nodes
//                        console.log(network_a.body.nodes[node]["id"])
                    }
                    if (edges.length == 1) {

                        let click = event.event.center
 /*
                        obj.setState({
                            panel_trigger_1: obj.state.panel_trigger_1 + 1
                        })
 */
                        console.log(network_a, edges)
                        console.log("GF2", obj.state.graph_features[network_a.body.edges[edges]["fromId"] + "," + network_a.body.edges[edges]["toId"]])
                        let metrics = obj.state.graph_features[network_a.body.edges[edges]["fromId"] + "," + network_a.body.edges[edges]["toId"]]
                        obj.setState({stats: {
                                number: metrics[0],
                                sjr: metrics[1],
                                if: metrics[2],
                                hi: metrics[3],
                            }})
                        console.log("GF2", obj.state.graph_features)

//                        console.log(click, edges)
                        console.log(network_a.body.edges[edges]["fromId"], network_a.body.edges[edges]["toId"])
                    }
                }
            }; //onClick={this.showWhereClicked}

            return (
                <div className="inner_container">
                    <div className="info_box" style={this.state.placedDiv}>
                        <div className="title">CHOSEN COLLAB INFO</div>
                        <div className="info_number field">
                            <div>Number of publications: </div>
                        </div>
                        <div className="info_sjr field">
                            <div>Expected SJR: {this.state.stats.sjr}</div>
                        </div>
                        <div className="info_if field">
                            <div>Expected impact factor: {this.state.stats.if}</div>
                        </div>
                        <div className="info_hi field">
                            <div>Expected h-index: {this.state.stats.hi}</div>
                        </div>
                    </div>
                    <Graph
                        graph={graph}
                        options={options}
                        events={events}
                        getNetwork={network => {
                            network_a = network
                            console.log("Network", network_a)
                        }}
                    />
                </div>
            );
        } else {
            return (
                <div className="inner_container" onClick={this.showWhereClicked}>

                </div>
            )
        }
    }
}

export default Network;
