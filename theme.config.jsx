import { useRouter } from 'next/router';
import styles from "./styles/Home.module.css"

export default {
    logo: ()=>{return(
      <>
    <img src="../../logo192.png" width="50vw"/>
    <p style={{"margin-left":"1vw"}}>At UC San Diego</p>
    </>
    )},

    useNextSeoProps() {
      const { asPath } = useRouter();
      if (asPath !== '/') {
        return {
          titleTemplate: '%s – ACM AI',
        };
      }
      return {
        titleTemplate: 'ACM AI Hack School',
      };
    },
    footer: {
      text: 'Made with 🧡 by ACM AI! Inspired by our friends at ACM Hack.',
    },
    search: {
      placeholder: 'Search',
    },
    sidebar: {
      toggleButton: true,
    }
  }