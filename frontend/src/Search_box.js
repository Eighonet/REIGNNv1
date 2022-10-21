import React from "react";
import { useEffect } from "react";
import './css/searchbox.css';
import './css/Search_box.css';
import { ReactDialogBox } from 'react-js-dialog-box'
import 'react-js-dialog-box/dist/index.css'

const REQUEST_URL = 'https://raw.githubusercontent.com/Eighonet/REIGNN_demo/main/front/data_s.json';
//const REQUEST_URL = 'https://jonasjacek.github.io/colors/data.json';

class Search_box extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            number_of_new_users: 0,
            name: null,
            surname: null,
            data: null,
            search: "",
            color: [],
            colors: [],
            dialogIsOpen: false,
            isListening: false,
            toSend: [],
            interests: null
        }
    }

    componentDidMount() {
        /*
         fetch(REQUEST_URL)
            .then(response => response.json())
            .then(data => {
                this.setState({data})
            })
        */
        let data = '[{"name": "Test", "id":123}]'
        this.setState({data})
    }



    // Select the wrapper and toggle class 'focus'
    onFocus = e => e.target.parentNode.parentNode.classList.add('focus');
    onBlur = e => e.target.parentNode.parentNode.classList.remove('focus');
    // Select item

    onClickItem(item) {
    this.state.colors.push(item)
    this.setState({
        search: "",
        color: item
    });
    }

    onDeleteClick() {
    this.setState({
        search: "",
        colors: []
    });
    }

    onClickAddResearcher() {
    this.setState({
        dialogIsOpen: true
    });
    }

    handleChangeName = (e) => {
        this.setState({
                name: e.target.value
        });
    }

    handleChangeSurname = (e) => {
        this.setState({
                surname: e.target.value
        });
    }

    closeBox = () => {
    this.setState({
        dialogIsOpen: false
    })
    }

    submitNewAuthor() {
        let authors = "Test Author", feature_vector = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        let item = {"name": this.state.name + " " + this.state.surname, "id": this.state.number_of_new_users}


        this.state.colors.push(item)

        this.setState({
            search: "",
            number_of_new_users: this.state.number_of_new_users - 1,
            color: item
        });
        this.closeBox()

        console.log(JSON.stringify({"name":authors, "feature_vector":feature_vector}))
        fetch("https://reignn.online:9998/add_author", {
            method: 'POST',
            body:JSON.stringify({"name":authors, "feature_vector":feature_vector})
        }).then(response => response.json())
            .then(data => {
                console.log(data)
            })

    }
    onInput(e, ws_search) {
        const onmessage = () => {
            ws.onmessage = (event) => {
//                console.log("Data received 3:", event.data);
                let data = event.data
                this.setState({data})
            };
        };
        let ws = ws_search
        ws.onopen = () => ws.send(e.target.value);

        onmessage();
        this.setState({ [e.target.id]: e.target.value });
    }

    onSubmitClick(ws_get) {
        let authors = [], authors_new = []

        for (let i=0; i<this.state.colors.length;i++) {
            if (parseInt(this.state.colors[i]["id"]) > 1) {
                authors.push(parseInt(this.state.colors[i]["id"]))
            } else {
                authors_new.push(parseInt(this.state.colors[i]["id"]))
            }
        }


        /*
        fetch("https://reignn.online/choose_authors", {
            method: 'POST',
            body:JSON.stringify({"authors":authors}),
            headers: {
                'Content-Type': 'application/json'
            }
        }).then(response => response.json())
            .then(data => {
                console.log(data)
                alert(data)
            })
         */

//        this.setState({isListening:true, toSend: authors})

                const onmessage = () => {
                    alert(message   )
                    ws_get.onmessage = (event) => {
                        let data = JSON.parse(JSON.parse(event.data))
                        console.log(data);
                        if ('author_edgelist' in data) {
                            fetch("https://reignn.online:9998/get_author_feature_vector", {
                                method: 'POST',
                                body:JSON.stringify({"authors":authors}),
                                headers: {
                                    'Content-Type': 'application/json'
                                }
                            }).then(response => response.json())
                                .then(data_interests => {
                                    this.setState({ interests: data_interests});
                                    this.props.getCollabsInfo(data, this.state.colors, this.state.interests)
                           //         this.props.getResearcherIds(this.state.colors, this.state.interests)
                                    console.log("Data interests", data_interests)
                                })
                        }
                    };
                };

                let message = JSON.stringify({"authors": authors})
                ws_get.onopen = () => ws_get.send(message);
                onmessage();
    }

render() {
    const ws_search = new WebSocket("wss://reignn.online:9998/ws_search_authors")
//    const ws_search = null
    const ws_get = new WebSocket("wss://reignn.online:9998/ws_predict")

    if (this.state.isListening == true) {

        const onmessage = () => {
            ws_get.onmessage = (event) => {
                let data = JSON.parse(event.data)
                console.log(data);
            };
        };

        let message = JSON.stringify({"authors": [1, 2, 3, 4, 5]})
        ws_get.onopen = () => ws_get.send(message);
        onmessage();
    }

    let { data, search, color } = this.state;

    if (!data) {
        return <p>Loading</p>
    }

    //        alert()
    data = JSON.parse(data)


    let filtered = data.filter(item => item.name.toLowerCase().includes(search.toLowerCase()));
    let team = []
    for (let i = 0; i < this.state.colors.length; i++) {
        let color = this.state.colors[i]
        team.push(<p className="result">
            <b>Member:</b>
            {color.name}
            <span className="box" style={{backgroundColor: color.hexString}}/>
        </p>)
    }

return (
    <div>
        {this.state.dialogIsOpen && (
            <>
                <ReactDialogBox
                    closeBox={this.closeBox}
                    modalWidth='40%'
                    headerBackgroundColor='white'
                    headerTextColor='black'
                    headerHeight='65'
                    closeButtonColor='black'
                    bodyBackgroundColor='white'
                    bodyTextColor='black'
                    bodyHeight='365px'
                    headerText='NEW RESEARCHER PROFILE'
                >
                    <form className="form">
                        <div className="form_inputs">
                            <div className="input_wrapper">
                                Name
                                <input className="new_user name"  onChange={ this.handleChangeName} type="name"/>
                            </div>
                            <div className="input_wrapper">
                                Surname
                                <input className="new_user surname" onChange={ this.handleChangeSurname} type="surname"/>
                            </div>

                        </div>
                        <div className="division_line"></div>
                        <h4>Research interests</h4>
                        <div className="checkboxes">
                            <div className="checkbox">
                                <input type="checkbox" name="Art"/>
                                <label htmlFor="Art">Art</label>
                            </div>

                            <div className="checkbox">
                                <input type="checkbox" name="Biology"/>
                                <label htmlFor="Biology">Biology</label>
                            </div>

                            <div className="checkbox">
                                <input type="checkbox" name="Business"/>
                                <label htmlFor="Business">Business</label>
                            </div>

                            <div className="checkbox">
                                <input type="checkbox" name="Chemistry"/>
                                <label htmlFor="Chemistry">Chemistry</label>
                            </div>

                            <div className="checkbox">
                                <input type="checkbox" name="Computer Science"/>
                                <label htmlFor="Computer Science">Computer Sc.</label>
                            </div>

                            <div className="checkbox">
                                <input type="checkbox" name="Economics"/>
                                <label htmlFor="Economics">Economics</label>
                            </div>

                            <div className="checkbox">
                                <input type="checkbox" name="Engineering"/>
                                <label htmlFor="Engineering">Engineering</label>
                            </div>

                            <div className="checkbox">
                                <input type="checkbox" name="Environmental Science"/>
                                <label htmlFor="Environmental Science">Environmental Sc.</label>
                            </div>

                            <div className="checkbox">
                                <input type="checkbox" name="Geography"/>
                                <label htmlFor="Geography">Geography</label>
                            </div>

                            <div className="checkbox">
                                <input type="checkbox" name="Geology"/>
                                <label htmlFor="Geography">Geology</label>
                            </div>

                            <div className="checkbox">
                                <input type="checkbox" name="History"/>
                                <label htmlFor="History">History</label>
                            </div>

                            <div className="checkbox">
                                <input type="checkbox" name="Materials Science"/>
                                <label htmlFor="Materials Science">Materials Science</label>
                            </div>

                            <div className="checkbox">
                                <input type="checkbox" name="Mathematics"/>
                                <label htmlFor="Mathematics">Mathematics</label>
                            </div>

                            <div className="checkbox">
                                <input type="checkbox" name="Medicine"/>
                                <label htmlFor="Medicine">Medicine</label>
                            </div>

                            <div className="checkbox">
                                <input type="checkbox" name="Philosophy"/>
                                <label htmlFor="Philosophy">Philosophy</label>
                            </div>

                            <div className="checkbox">
                                <input type="checkbox" name="Physics"/>
                                <label htmlFor="Physics">Physics</label>
                            </div>

                            <div className="checkbox">
                                <input type="checkbox" name="Politics Science"/>
                                <label htmlFor="Politics Science">Politics Science</label>
                            </div>

                            <div className="checkbox">
                                <input type="checkbox" name="Psychology"/>
                                <label htmlFor="Psychology">Psychology</label>
                            </div>
                            <div className="checkbox">
                                <input type="checkbox" name="Sociology"/>
                                <label htmlFor="Sociology">Sociology</label>
                            </div>
                        </div>
                        <div className="button_apply" onClick={() => this.submitNewAuthor()}>Apply</div>
                    </form>


                </ReactDialogBox>
            </>
        )}

        <div className="search_block">
            <div className="wrapper">
                    <div className="search">
                        <input
                            id="search"
                            value={this.state.search}
                            placeholder="Type researcher name"
                            onChange={(e) => {this.onInput(e, ws_search)}}
                            onFocus={this.onFocus}
                            onBlur={this.onBlur}
                            autoComplete="off"
                        />
                        <i class="fas fa-search"></i>
                        <div className="newResearcher_button" onClick={() => this.onClickAddResearcher()}><div className="symbol_add">+</div></div>
                    </div>
                    {search.length > 1 && filtered.length > 0 && (
                        <ul className="list">
                            {filtered.slice(0, 10).map(item => (
                                <li onClick={() => this.onClickItem(item)}>{item.name}</li>
                            ))}
                        </ul>
                    )}
            </div>
        </div>
        <div className="team">
            <div className="team_title">Team</div>
            <div className="inner_team">
                {team}
            </div>
        </div>
        <div className="buttons">
            <div className="drop_button" onClick={() => this.onDeleteClick()}><div className="symbol">×</div></div>
            <div className="confirm_button" onClick={() => this.onSubmitClick(ws_get)}><div className="symbol">✓</div></div>
        </div>
    </div>
)
}

};

export default Search_box;

