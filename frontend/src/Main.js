import React from "react";
import { Scrollbar } from "react-scrollbars-custom";
import './css/Main.css';
import Search_box from "./Search_box";
import Team from "./Team";
import Stats from "./Stats";
import Network from "./Network";

class Main extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            team: [],
            interests: [],
            edges: [],
            graph: []
        }
    }

    getResearcherIds(node_ids, data_interests) {
        console.log("features 2", data_interests)
        this.setState({
            team: node_ids,
            interests: data_interests
        })
    }

    getCollabsInfo(graph, node_ids, data_interests) {
        console.log("graph", graph, node_ids, data_interests)
        this.setState({
            graph: graph,
            team: node_ids,
            interests: data_interests
        })
    }

    render() {

        return(
            <div>
                <div className="main">
                    <div className="left">
                        <div className="searchbox_wrapper">
                            <div class="search_title">REIGNN: Fast and Efficient R&D Team Management Service</div>
                            <Search_box getCollabsInfo={(e3, e2, e1) => this.getCollabsInfo(e3, e2, e1)}/>
                        </div>
                    </div>
                    <div className="center">
                        <Network node_ids={this.state.team} graph={this.state.graph}/>
                    </div>
                    <div className="right">

                        <Stats interests={this.state.interests} graph={this.state.graph}/>
                        <div className="space"></div>
                    </div>
                </div>
                <a href="https://github.com/Eighonet/REIGNN_demo">
                    <div className="github">
                        <div className="gh_icon"></div>
                        <div className="gh_link_name">GITHUB</div>
                    </div>
                </a>
            </div>
        )
    }
}

export default Main;